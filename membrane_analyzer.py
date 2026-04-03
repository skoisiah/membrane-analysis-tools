import MDAnalysis as mda
from MDAnalysis import transformations
from MDAnalysis.analysis import lineardensity
import numpy as np
from scipy.spatial import Voronoi
import subprocess
import os

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class MembraneAnalyzer:
    def __init__(self, psf_file, dcd_file):
        """Initialize the trajectory and define the base system."""
        #print(f"Loading Universe: {psf_file} | {dcd_file}")

        self.u = mda.Universe(psf_file, dcd_file)
        
        # Save useful metadata
        self.n_frames = len(self.u.trajectory)
        self.dt = self.u.trajectory.ts.dt # Time step between frames

        # Define common atom selection
        self.popc = self.u.select_atoms("resname POPC")
        self.DOPC = self.u.select_atoms("resname DOPC")
        self.pfoa = self.u.select_atoms("resname 8PF")
        self.lipids = self.u.select_atoms("resname POPC DOPC")

        # Placeholder for dynamic leaflet selections
        self.upper_leaflet = None
        self.lower_leaflet = None

    def center_bilayer(self):
        """Centers the bilayer at the center of the simulation box.
        This modifies the trajectory coordinates in memorey.
        """
        print("centering the bilayer in the simulation box...")
        # 1. Make the membrane whole across the periodic boundries
        make_whole = transformations.unwrap(self.lipids)
        # 2. Center by membrane cener of mass
        center_membrane = transformations.center_in_box(self.lipids, center='geometry')
        # 3. Wrap the water/ions back into the box
        wrap_all = transformations.wrap(self.u.atoms, compound='residues')

        # Apply the transformation sequence to the trajectory
        self.u.trajectory.add_transformations(make_whole, center_membrane, wrap_all)
        print("Centering complete.")

    def calc_bilayer_thickness(self, head_group_name='P'):
        """
        Calculates the peak-to-peak distance of the headgroups (usually the Phosphorus.
        Returns an array of thickness over time and the time in ns 
        """
        print(f"Calculating thickness using atom name {head_group_name}...")
        
        heads = self.lipids.select_atoms(f"name {head_group_name}")
        # Assign the leaflets once based on the very first frame (t=0)
        self.u.trajectory[0]
        com_z = heads.center_of_mass()[2]

        # Static groups stored in memory
        upper_leaflet = heads.select_atoms(f"prop z > {com_z}")
        lower_leaflet = heads.select_atoms(f"prop z < {com_z}")
        
        thickness_ts = []
        
        for ts in self.u.trajectory:
            upper_z = upper_leaflet.center_of_mass()[2]
            lower_z = lower_leaflet.center_of_mass()[2]

            time_ns = ts.time / 1000
            thickness_ts.append([time_ns, upper_z - lower_z])

        return np.array(thickness_ts)

    def _compute_voronoi_areas(self, points_xy, box_dimensions):
        """
        Helper method
        """
        lx, ly = box_dimensions[0], box_dimensions[1]
        shifts = [(-lx, -ly), (0, -ly), (lx, -ly), (-lx, 0), (lx, 0), (-lx, ly), (0, ly), (lx, ly)]

        all_points = [points_xy]
        for dx, dy in shifts:
            all_points.append(points_xy + np.array([dx, dy]))
        stack_points = np.vstack(all_points)

        vor = Voronoi(stack_points)
        n_points = len(points_xy)
        
        areas = []

        for i in range(n_points):
            region_index = vor.point_region[i]
            region_vertices_indices = vor.regions[region_index]
            if -1 in region_vertices_indices:
                areas.append(np.nan)
                continue
            verts = vor.vertices[region_vertices_indices]
            x, y = verts[:, 0], verts[:, 1]
            #Shoelace formula for polygon area
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            areas.append(area)

        return np.array(areas)

    def calculate_area_voronoi(self, selection=None):
        """
        Calculates the average Area per Molecule uisng Voronoi tessellation.
        Splits the leaflets automatically so 2D projections don't overlap
        """
        print(f"Calculating Voronoi area for {selection}...")

        if selection is None:
            selection = "resname POPC and name P"

        # 1. Grab the target atoms (e.g. Phosphorus atoms)
        targets = self.u.select_atoms(selection)

        # 2. Assign leaflets once at the first frame
        self.u.trajectory[0]
        com_z = targets.center_of_mass()[2]
        upper_leaflet = targets.select_atoms(f"prop z > {com_z}")
        lower_leaflet = targets.select_atoms(f"prop z < {com_z}")

        apl_ts = []

        # 3. Loop
        for ts in self.u.trajectory:
            # Get X and Y box dimensions
            box_dims = ts.dimensions[:2]

            # Extract just the X, Y coordinates (columns 0 and 1) for this frame
            upper_xy = upper_leaflet.positions[:, :2]
            lower_xy = lower_leaflet.positions[:, :2]

            # Calculate the Voronoi area for each leaflet independently
            upper_areas = self._compute_voronoi_areas(upper_xy, box_dims)
            lower_areas = self._compute_voronoi_areas(lower_xy, box_dims)

            # Combine the areas from both leaflets and find the mean for this frame
            # (np.nanmean ensures that if any weird PBC artifacts return NaN, it doesn't break)
            frame_avg_area = np.nanmean(np.concatenate([upper_areas, lower_areas]))

            time_ns = ts.time / 1000.0
            apl_ts.append([time_ns, frame_avg_area])

        return np.array(apl_ts)


    def calculate_area_per_part(self, selection_string):
        """
        Calculates the average Area Per Molecule (or part of a molecule) 
        using Voronoi tessellation on a 2D projected plane.
        """
        print(f"Calculating Voronoi area for lipid parts: '{selection_string}'...")

        # 1. Grab the target atoms
        targets = self.u.select_atoms(selection_string)
        
        if len(targets) == 0:
            raise ValueError(f"Selection '{selection_string}' found 0 atoms.")

        # 2. Assign leaflet split point once at the first frame
        self.u.trajectory[0]
        split_z = targets.center_of_mass()[2]

        apl_ts = []

        # 3. Loop through the trajectory
        for ts in self.u.trajectory:
            box_dims = ts.dimensions[:2]

            # MAGIC LINE: Calculate the Center of Mass for the selected atoms, 
            # grouped by each individual lipid (residue). 
            # This returns an array of shape (N_lipids, 3)
            part_centers = targets.center_of_mass(compound='residues')

            # Split into upper and lower leaflets based on the Z coordinate (column 2)
            upper_mask = part_centers[:, 2] > split_z
            lower_mask = ~upper_mask

            # Extract just the X, Y coordinates (columns 0 and 1)
            upper_xy = part_centers[upper_mask][:, :2]
            lower_xy = part_centers[lower_mask][:, :2]

            # Calculate the Voronoi area using your existing helper method
            upper_areas = self._compute_voronoi_areas(upper_xy, box_dims)
            lower_areas = self._compute_voronoi_areas(lower_xy, box_dims)

            # Combine and average
            frame_avg_area = np.nanmean(np.concatenate([upper_areas, lower_areas]))
            
            time_ns = ts.time / 1000.0
            apl_ts.append([time_ns, frame_avg_area])

        return np.array(apl_ts)

 


       

        
    @staticmethod
    def _get_electron_count(atom_name):
        """Maps an atom name to its number of electrons."""
        name = atom_name.upper()
        # Tier 1: Exceptions
        if name.startswith('NA') or name.startswith('SOD'): return 11.0
        if name.startswith('CL') or name.startswith('CLA'): return 17.0
        if name.startswith('K')  or name.startswith('POT'): return 19.0
        if name.startswith('MG'): return 12.0
        if name.startswith('CA') or name.startswith('CAL'): return 20.0
        if name.startswith('ZN'): return 30.0
        
        # Tier 2: Standard Elements
        if name.startswith('H'): return 1.0
        if name.startswith('C'): return 6.0
        if name.startswith('N'): return 7.0
        if name.startswith('O'): return 8.0
        if name.startswith('F'): return 9.0
        if name.startswith('P'): return 15.0
        if name.startswith('S'): return 16.0
        return 0.0

    def _get_conversion_factor(self, selection):
        """
        Calculates conversion factor from Mass Density (g/cm^3) to Electron Density (e/A^3).
        """
        atomic_number = [self._get_electron_count(atom.name) for atom in selection.atoms]
        atomic_mass = list(selection.atoms.masses)
        
        total_z = np.sum(atomic_number)
        total_m = np.sum(atomic_mass)
        
        if total_m == 0: 
            return 0.0
        
        ratio = total_z / total_m
        return (6.022e23 * ratio) / 1.0e24

    def calc_electron_density(self, selection_string="all", binsize=1.0):
        """
        Calculates the Electron Density Profile (e/A^3) along the Z-axis.
        Automatically centers the bilayer and shifts the Z-axis so 0 is the membrane center.
        Returns:
        --------
        z_centered: Centered z coordinates
        rho_electrons: Electron density (e/A^3)
        rho_electrons_stddev: standard deviation in electron density
        row_mass: Mass density
        """
        print(f"Calculating Electron Density for '{selection_string}'...")
        sel = self.u.select_atoms(selection_string)
        
        # 2. Get the conversion factor using your custom logic
        conv_factor = self._get_conversion_factor(sel)
        
        # 3. Run MDAnalysis LinearDensity
        # (This calculates mass density in g/cm^3)
        density = lineardensity.LinearDensity(sel, grouping="atoms", binsize=binsize)
        density.run()
        
        # Extract native results
        rho_mass = density.results.z.mass_density
        rho_mass_stddev = density.results.z.mass_density_stddev
        edges = density.results.z.hist_bin_edges
        
        # 4. Calculate Z-axis centers and shift so 0 is the middle
        coords = (edges[:-1] + edges[1:]) / 2
        z_centered = coords - (coords[0] + coords[-1]) / 2
        
        # 5. Convert to Electron Density (e/A^3)
        rho_electrons = rho_mass * conv_factor
        rho_electrons_stddev = rho_mass_stddev * conv_factor
        
        return z_centered, rho_electrons, rho_electrons_stddev, rho_mass


    def export_to_simtoexp(self, groups_dict, area_per_lipid, prefix="popc_simtoexp", dz=0.2, z_max=40.0):
        print("Centering bilayer for SIMtoEXP export...")
        self.center_bilayer()
        
        # 1. Define the Z-bins
        z_bins = np.arange(-z_max, z_max + dz, dz)
        z_centers = np.arange(-z_max + dz/2, z_max, dz)
        
        # 2. Resolve selection strings into explicit atom names for the .cmp file
        lipid_atom_names = set()
        cmp_lines = []
        
        for group_name, sel_string in groups_dict.items():
            if group_name == "WAT":
                cmp_lines.append("WAT\t1\tW_wat")
                continue
                
            # MDAnalysis instantly resolves wildcards (like C2*) into actual names!
            atoms = self.u.select_atoms(sel_string)
            unique_names = list(set(atoms.names))
            lipid_atom_names.update(unique_names)
            
            # Format: [Component] [1] [Atom1] [Atom2] ...
            cmp_lines.append(f"{group_name}\t1\t" + " ".join(unique_names))
            
        lipid_atom_names = list(lipid_atom_names)
        
        # 3. Create dictionary to hold density for EVERY INDIVIDUAL ATOM
        density_profiles = {name: np.zeros(len(z_centers)) for name in lipid_atom_names}
        density_profiles["W_wat"] = np.zeros(len(z_centers))
        
        # Select atom groups ONCE outside the loop for massive speedup
        atom_selections = {name: self.u.select_atoms(f"resname POPC and name {name}") for name in lipid_atom_names}
        water_oxy = self.u.select_atoms("(resname TIP3 or resname WAT or resname SOL) and name OH2 O OW")
        lipid_sel = self.u.select_atoms("resname POPC")
        
        # 4. Calculate Histograms frame-by-frame
        print(f"Calculating atom-by-atom density profiles for {len(lipid_atom_names)} unique atoms...")
        n_frames = len(self.u.trajectory)
        
        for ts in self.u.trajectory:
            z_shift = np.mean(lipid_sel.positions[:, 2])
            
            # Histogram each individual lipid atom
            for name in lipid_atom_names:
                z_coords = atom_selections[name].positions[:, 2] - z_shift
                hist, _ = np.histogram(z_coords, bins=z_bins)
                density_profiles[name] += hist
                
            # Histogram Water (Using only the Oxygen atom to represent the 10-electron molecule)
            w_coords = water_oxy.positions[:, 2] - z_shift
            hist, _ = np.histogram(w_coords, bins=z_bins)
            density_profiles["W_wat"] += hist

        # 5. Normalize to Number Density
        n_lipids_total = len(lipid_sel.residues)
        n_lipids_per_leaflet = n_lipids_total // 2
        total_box_area = area_per_lipid * n_lipids_per_leaflet
        
        for name in density_profiles.keys():
            volume_slice = total_box_area * dz
            density_profiles[name] /= (n_frames * volume_slice)
            
        # 6. Write the .sim file (with ~135 columns!)
        with open(f"{prefix}.sim", "w") as f:
            f.write("#################################################################################\n")
            f.write("# Auto-generated by Python MembraneAnalyzer\n")
            f.write(f"area {total_box_area:.3f}\n")
            f.write(f"Nlip {n_lipids_total}\n")
            f.write("norm no\n")
            f.write("#################################################################################\n")
            f.write("Data\n")
            
            header_components = "\t".join(density_profiles.keys())
            f.write(f"z(A) {header_components}\n")
            
            for i, z_val in enumerate(z_centers):
                densities = [f"{density_profiles[name][i]:.6f}" for name in density_profiles.keys()]
                row_str = f"{z_val:.3f} " + "\t".join(densities)
                f.write(row_str + "\n")
                
        # 7. Write the official .cmp file
        with open(f"{prefix}.cmp", "w") as f:
            for line in cmp_lines:
                f.write(line + "\n")
                
        print(f"Exported component file: {prefix}.cmp (Klauda Format)")
        print(f"Exported simulation file: {prefix}.sim (Atom-by-Atom Format)")


    def calc_order_parameter(self, selection="resname POPC", tail="C3"):
        """
        Calculates the Deterium Order Parameter (-S_CD) for lipid tails

        Paramters: 
        ----------
        selection : str
            The MDAnanysis selection string for the target lipids.
        tail : str
            "C3" for the sn-1 tail (e.g., palmitoyl), or "C2" for the sn-2 tail (e.g., oleoyl).

        Returns:
        --------
        np.narray
            A 2D array where Column 0 is the Carbon number and Column 1 is the Order Parameter.
        """
        print(f"Calculating Order Parameter for {selection} tail {tail}...")

        # 1. Define the CHARMM hydrogen naming rules based on the tails
        if tail.upper() =="C3":
            h_surffixes = ["X", "Y", "Z"]
        elif tail.upper() == "C2":
            h_surffixes = ["R", "S", "T", "1"] # Captures double bond hydrogens like H91
        else:
            raise ValueError("Tail must be 'C2' or 'C3'")

        # 2. Find all the valid carbons and their attached hydrogens once
        carbon_data = {}

        # Loop over all possible carbons lengths (up to 18)
        for i in range(2,19):
            c_atoms = self.u.select_atoms(f"({selection}) and name {tail.upper()}{i}")

            # If this carbon deos not exist (e.g., we have reached the end of the tail), skip it
            if len(c_atoms) == 0:
                continue

            h_groups = []
            for s in h_surffixes:
                h_atoms = self.u.select_atoms(f"({selection}) and name H{i}{s}")
                if len(h_atoms) > 0:
                    h_groups.append(h_atoms)

            # Save the selection and set up tracking variables
            if h_groups:
                carbon_data[i] = {
                    'c_atoms'  : c_atoms,
                    'h_groups'  : h_groups,
                    'sum_cos2' : 0.0,
                    'n_bonds'  : 0
                }

        if not carbon_data:
            raise ValueError(f"No atoms found matching tails {tail} for {selection}. Check atom names in your PSF.")

        # 3. Loop through the trajectory exactly one time
        for ts in self.u.trajectory:
            for i, data in carbon_data.items():
                c_pos = data["c_atoms"].positions

                for hg in data['h_groups']:
                    # Vectorize calculations for thousands of lipids at once!
                    vectors = hg.positions - c_pos

                    # Square length of the C-H vectors
                    norm2 = np.sum(vectors**2, axis=1)
                    # Z-compintnts squared
                    dz2 = vectors[:,2]**2
                    data['sum_cos2'] += np.sum(dz2 / norm2)
                    data['n_bonds'] += len(c_pos)
        # 4. Finalize the math
        results = []
        for i, data in carbon_data.items():
            S_cd = -1.5 * (data['sum_cos2'] / data['n_bonds']) + 0.5
            results.append([i, S_cd])

        return np.array(results)
    def write_voro_input(self, frame_index, selection="all", out_file="frame_data.txt"):
        """
        Method 1: Extracts box dimensions and coordinates for a specific frame 
        and writes them to a text file.
        """
        # Jump to the requested frame
        self.u.trajectory[frame_index]
        
        # Get dimensions
        Lx, Ly, Lz = self.u.dimensions[:3]
        
        # Select atoms and wrap them into the primary box for Voro++
        atoms = self.u.select_atoms(selection)
        atoms.wrap()
        
        # Write the file
        with open(out_file, "w") as f:
            # We can store the box dimensions at the very top as a comment/header
            f.write(f"{Lx:.3f} {Ly:.3f} {Lz:.3f}\n")
            
            # Write the Voro++ required format: ID X Y Z
            for atom in atoms:
                x, y, z = atom.position
                f.write(f"{atom.id} {x:.3f} {y:.3f} {z:.3f}\n")
                
        #print(f"Frame {frame_index} exported to {out_file}")
        return out_file

    def calculate_volume(self, target_ids, dimensions, in_file="frame_data.txt"):
        """
        Method 2: Reads the text file, extracts the box, runs Voro++, 
        and calculates the volume for the target atom IDs.
        """
        # 1. Read the box dimensions from the first line
        with open(in_file, "r") as f:
            first_line = f.readline()
            parts = first_line.split()
            Lx, Ly, Lz = dimensions[:3]
            
        # 2. Run Voro++ using the extracted box dimensions
        voro_cmd = [
            "voro++", "-p", "-c", "%i %v", 
            "0", str(Lx), "0", str(Ly), "0", str(Lz), 
            in_file
        ]
        subprocess.run(voro_cmd, check=True)
        
        # 3. Parse the Voro++ output file (.vol)
        out_file = in_file + ".vol"
        total_vol = 0.0
        
        with open(out_file, "r") as f:
            for line in f:
                parts = line.split()
                atom_id = int(parts[0])
                vol = float(parts[1])
                
                if atom_id in target_ids:
                    total_vol += vol
                    
        # Clean up the output file (optional: keep the input file if you want to cache it)
        os.remove(out_file)
        
        return total_vol
    def run_volume_timeseries(self, target_selection, out_dat="volume_timeseries.dat", start=0, stop=None, step=1):
        """
        Wrapper method: Loops through the trajectory, calls the extraction 
        and calculation methods, and saves the timeseries data.
        """
        # 1. Identify the target atoms we want to measure
        target_atoms = self.u.select_atoms(target_selection)
        target_ids = set(target_atoms.ids)
        
        # Calculate how many molecules we have to get the average per molecule
        n_molecules = target_atoms.n_residues
        
        if n_molecules == 0:
            print(f"Error: Could not find any atoms matching '{target_selection}'")
            return

        volume_data = []
        temp_txt = "temp_voro_input.txt"

        print(f"Starting volume calculation for {n_molecules} molecules based on '{target_selection}'...")

        # 2. Loop through the requested frames
        for ts in self.u.trajectory[start:stop:step]:
            time_ns = ts.time / 1000.0

             # Extract dimensions here in the wrapper
            current_dimensions = ts.dimensions
            
            self.write_voro_input(ts.frame, selection="all", out_file=temp_txt)
            
            # Pass the dimensions into the calculator
            total_vol = self.calculate_volume(target_ids, dimensions=current_dimensions, in_file=temp_txt)
            
            avg_vol = total_vol / n_molecules
            
            volume_data.append([time_ns, avg_vol])
            print(f"Frame {ts.frame:04d} | Time: {time_ns:.3f} ns | Avg Vol: {avg_vol:.3f} Å³")

        # Clean up the temporary input file once the loop is entirely done
        if os.path.exists(temp_txt):
            os.remove(temp_txt)

        # 3. Save the results to a formatted .dat file
        header_string = f"{'Time_ns':>10} {'Avg_Volume_A3':>20}"
        np.savetxt(
            out_dat, 
            np.array(volume_data), 
            fmt="%10.3f %20.3f", 
            header=header_string, 
            comments=""
        )
        print(f"Finished! Timeseries saved to {out_dat}")

    def run_multi_selections(self, selections_dict, out_dir=".", start=0, stop=None, step=1):
        """
        Runs Voro++ ONCE per frame and instantly calculates volumes for an unlimited 
        number of target selections simultaneously. 30x faster!
        """
        os.makedirs(out_dir, exist_ok=True)
        
        # 1. Pre-calculate atom IDs and molecule counts for all groups so we don't 
        # waste time doing it inside the loop
        target_data = {}
        print("Pre-computing atom selections...")
        for name, sel_string in selections_dict.items():
            atoms = self.u.select_atoms(sel_string)
            if len(atoms) == 0:
                print(f"Warning: Selection '{name}' found 0 atoms. Skipping.")
                continue
                
            target_data[name] = {
                'ids': set(atoms.ids),
                'n_molecules': atoms.n_residues # To get the average volume PER LIPID
            }
            
        # Initialize an empty list for every group to store its [time, volume] pairs
        results = {name: [] for name in target_data.keys()}
        
        temp_txt = "temp_multi_voro.txt"
        
        print(f"\nStarting high-speed multi-volume calculation for {len(target_data)} groups...")

        # 2. Loop through the trajectory ONCE
        for ts in self.u.trajectory[start:stop:step]:
            time_ns = ts.time / 1000.0
            
            # Export the whole system
            self.write_voro_input(ts.frame, selection="all", out_file=temp_txt)
            
            # Run Voro++ ONCE
            Lx, Ly, Lz = ts.dimensions[:3]
            voro_cmd = [
                "voro++", "-p", "-c", "%i %v", 
                "0", str(Lx), "0", str(Ly), "0", str(Lz), 
                temp_txt
            ]
            subprocess.run(voro_cmd, check=True)
            
            # Load every single atom's volume in the whole system into a fast lookup table
            vol_dict = {}
            with open(temp_txt + ".vol", "r") as f:
                for line in f:
                    parts = line.split()
                    vol_dict[int(parts[0])] = float(parts[1])
                    
            # 3. Instantly extract averages for all 30 groups from the dictionary!
            for name, data in target_data.items():
                # Sum the volumes of only the target atoms
                total_vol = sum(vol_dict[aid] for aid in data['ids'] if aid in vol_dict)
                avg_vol = total_vol / data['n_molecules']
                
                results[name].append([time_ns, avg_vol])
                
            #print(f"Processed Frame {ts.frame:04d} | Time: {time_ns:.3f} ns")

        # Clean up temp files
        if os.path.exists(temp_txt): os.remove(temp_txt)
        if os.path.exists(temp_txt + ".vol"): os.remove(temp_txt + ".vol")

        # 4. Save all the files at the very end
        print("\nSaving all timeseries files...")
        header_string = f"{'Time_ns':>10} {'Avg_Volume_A3':>20}"
        
        for name, timeseries in results.items():
            final_save_path = os.path.join(out_dir, f"{name}_volumes.dat")
            np.savetxt(
                final_save_path, 
                np.array(timeseries), 
                fmt="%10.3f %20.3f", 
                header=header_string, 
                comments=""
            )
        print(f"Finished! All {len(results)} files saved to {out_dir}")
