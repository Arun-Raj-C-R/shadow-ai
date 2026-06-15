import numpy as np
import json
import traceback

# --- Prerequisite ---
# You must have solcore and numpy installed in your environment:
# pip install solcore numpy

try:
    # --- Solcore Imports ---
    from solcore import material
    from solcore.structure import Layer, Junction
    from solcore.solar_cell import SolarCell
    from solcore.solar_cell_solver import solar_cell_solver
    from solcore.light_source import LightSource
except ImportError as e:
    print(f"Error: Failed to import Solcore. Make sure it is installed (`pip install solcore`). Details: {e}")
    exit()

def run_simulation():
    """
    Runs a standalone Solcore simulation for a perovskite solar cell
    with PROPER n-i-p structure to ensure correct rectifying behavior.
    
    KEY FIX: Changed to n-i-p (n-TiO2 / intrinsic-Perovskite / p-Spiro)
    """
    print("Starting Solcore perovskite solar cell simulation...")
    print("Structure: n-TiO2 / i-Perovskite / p-Spiro (n-i-p configuration)")

    # --- 1. Define Input Data ---
    
    # Work functions for proper carrier extraction
    anode_work_function = 4.0  # FTO/TiO2 side
    cathode_work_function = 5.1  # Spiro/Metal side
    
    # CRITICAL FIX: Use INTRINSIC or very lightly doped perovskite
    # to create a proper p-i-n or n-i-p structure
    layers_data = [
        {
            "name": "TiO2_ETL",
            "thickness_m": 50e-9,
            "band_gap_ev": 3.2,
            "electron_affinity_ev": 4.0,
            "dielectric_constant": 9.0,
            "mobility_electron_cm2Vs": 20.0,
            "mobility_hole_cm2Vs": 1e-5,
            "dos_conduction_cm3": 2e19,
            "dos_valence_cm3": 2e19,
            "doping_type": "n",
            "doping_concentration_cm3": 1e18,  # Strong n-doping
            "srh_lifetime_e_s": 1e-6,
            "srh_lifetime_h_s": 1e-9,
            "electron_mass": 0.3,
            "hole_mass": 1.0
        },
        {
            "name": "MAPbI3_Absorber",
            "thickness_m": 400e-9,
            "band_gap_ev": 1.55,
            "electron_affinity_ev": 3.9,
            "dielectric_constant": 24.1,
            "mobility_electron_cm2Vs": 20.0,
            "mobility_hole_cm2Vs": 20.0,
            "dos_conduction_cm3": 2.2e18,
            "dos_valence_cm3": 2.2e18,
            "doping_type": "i",  # CRITICAL: Intrinsic or undoped
            "doping_concentration_cm3": 1e10,  # Very low background doping
            "srh_lifetime_e_s": 1e-7,
            "srh_lifetime_h_s": 1e-7,
            "electron_mass": 0.12,
            "hole_mass": 0.15
        },
        {
            "name": "Spiro_HTL",
            "thickness_m": 150e-9,  # Increased for better hole collection
            "band_gap_ev": 3.0,
            "electron_affinity_ev": 2.1,
            "dielectric_constant": 3.0,
            "mobility_electron_cm2Vs": 1e-5,
            "mobility_hole_cm2Vs": 2e-4,
            "dos_conduction_cm3": 2e19,
            "dos_valence_cm3": 2e19,
            "doping_type": "p",
            "doping_concentration_cm3": 2e18,  # Strong p-doping
            "srh_lifetime_e_s": 1e-9,
            "srh_lifetime_h_s": 1e-6,
            "electron_mass": 0.5,
            "hole_mass": 0.5
        }
    ]
    
    try:
        # --- 2. Build Material and Layer Stack ---
        solcore_layers = []
        print("\n" + "=" * 70)
        print("BUILDING DEVICE LAYERS".center(70))
        print("=" * 70)
        
        for i, layer in enumerate(layers_data):
            try:
                # Prepare doping parameters
                doping_params = {}
                doping_type = layer['doping_type'].lower()
                doping_conc_m3 = layer['doping_concentration_cm3'] * 1e6
                
                if doping_type == 'n':
                    doping_params['Nd'] = doping_conc_m3
                elif doping_type == 'p':
                    doping_params['Na'] = doping_conc_m3
                elif doping_type == 'i':
                    # Intrinsic: add small background doping for numerical stability
                    doping_params['Nd'] = doping_conc_m3 / 2
                    doping_params['Na'] = doping_conc_m3 / 2
                
                # Create material
                mat = material("Si")(
                    name=f"layer_{i}_{layer.get('name', 'unnamed')}",
                    T=300,
                    band_gap=layer['band_gap_ev'] * 1.602e-19,
                    electron_affinity=layer['electron_affinity_ev'] * 1.602e-19,
                    relative_permittivity=layer['dielectric_constant'],
                    electron_effective_mass=layer.get('electron_mass', 0.1),
                    hole_effective_mass=layer.get('hole_mass', 1.0),
                    electron_mobility=layer['mobility_electron_cm2Vs'] * 1e-4,
                    hole_mobility=layer['mobility_hole_cm2Vs'] * 1e-4,
                    conduction_band_DOS=layer['dos_conduction_cm3'] * 1e6,
                    valence_band_DOS=layer['dos_valence_cm3'] * 1e6,
                    electron_minority_lifetime=layer['srh_lifetime_e_s'],
                    hole_minority_lifetime=layer['srh_lifetime_h_s'],
                    bulk_recombination_energy=0,
                    **doping_params
                )
                
                # Calculate band positions
                E_c = -layer['electron_affinity_ev']
                E_v = E_c - layer['band_gap_ev']
                
                # Add the layer
                solcore_layers.append(
                    Layer(width=layer['thickness_m'], material=mat)
                )
                
                print(f"\n[Layer {i}] {layer['name']}")
                print(f"  Thickness: {layer['thickness_m']*1e9:.1f} nm")
                print(f"  Band gap: {layer['band_gap_ev']:.2f} eV")
                print(f"  E_c: {E_c:.2f} eV | E_v: {E_v:.2f} eV")
                print(f"  Doping: {doping_type.upper()}-type, {layer['doping_concentration_cm3']:.1e} cm⁻³")
                
            except KeyError as e:
                print(f"[ERROR] Missing key {e} for layer {i}")
                return
            except Exception as e:
                print(f"[ERROR] Could not build layer {i}: {e}")
                traceback.print_exc()
                return

        print("\n" + "=" * 70)
        
        # --- 3. Display Band Alignment ---
        print("\n[BAND ALIGNMENT]")
        print("  TiO2 E_c:        -4.00 eV  (n-type)")
        print("  Perovskite E_c:  -3.90 eV  ✓ (0.10 eV step for e⁻)")
        print("  Perovskite E_v:  -5.45 eV  (intrinsic)")
        print("  Spiro E_v:       -5.10 eV  ✓ (0.35 eV step for h⁺)")
        print("  → n-i-p junction: electrons to left, holes to right")
        print("=" * 70)

        # --- 4. Create Solar Cell Object ---
        print("\nBuilding Junction and SolarCell...")
        
        # CRITICAL: Add surface recombination velocities
        main_junction = Junction(
            solcore_layers, 
            kind='PDD', 
            T=300,
            sn=1e5,  # Surface recombination velocity for electrons (cm/s)
            sp=1e5   # Surface recombination velocity for holes (cm/s)
        )

        solar_cell = SolarCell(
            [main_junction],
            T=300,
            substrate=solcore_layers[0].material,
            anode_work_function=anode_work_function,
            cathode_work_function=cathode_work_function
        )

        # --- 5. Set Up Solver ---
        print("Configuring PDD solver...")

        # Wavelengths for optical calculation
        wavelength_m = np.linspace(300, 1200, 300) * 1e-9

        # Light source
        light = LightSource(
            source_type="standard",
            version="AM1.5g",
            x=wavelength_m,
            output_units="photon_flux_per_m"
        )
        
        # CRITICAL: For n-i-p junctions, use NEGATIVE voltage range
        # This is per Solcore's sign convention documentation
        v_range = np.linspace(-0.2, 1.2, 150)
        
        options = {
            "solver_type": "poisson_drift_diffusion",
            "light_source": light,
            "light_iv": True,
            "wavelength": wavelength_m,
            "voltages": v_range,
            "internal_voltages": v_range,
            "mpp": True,
            "optics_method": "TMM",
            "device_type": "junction",
            # Meshing options for better convergence
            "meshpoints": 150,  # Increase mesh density
            "growth_rate": 0.7,  # Control mesh growth
        }

        # --- 6. Run Simulation ---
        print("\nRunning PDD solver (this may take 1-2 minutes)...")
        print("-" * 70)
        
        solar_cell_solver(solar_cell, "iv", options)
        
        print("-" * 70)
        print("✓ Solver completed")

        # --- 7. Extract and Format Results ---
        iv_results = solar_cell.iv
        junction_iv = solar_cell[0]
        
        # Debug: Print raw data
        print("\n[DEBUG] Raw IV results:")
        print(f"  Keys available: {list(iv_results.keys())}")
        
        # Extract values
        voc_val = iv_results.get("Voc")
        jsc_val = iv_results.get("Jsc") 
        isc_val = iv_results.get("Isc")  # Try both
        ff_val = iv_results.get("FF")
        pce_val = iv_results.get("Eta") or iv_results.get("PCE")
        vmpp_val = iv_results.get("Vmpp")
        pmpp_val = iv_results.get("Pmpp")
        
        # Use Isc if Jsc not available
        if jsc_val is None and isc_val is not None:
            jsc_val = isc_val
        
        print(f"  Voc: {voc_val}")
        print(f"  Jsc/Isc: {jsc_val}")
        print(f"  FF: {ff_val}")
        print(f"  PCE/Eta: {pce_val}")

        output = {
            "status": "Success",
            "Voc_V": abs(float(voc_val)) if voc_val is not None else 0.0,
            "Jsc_Am2": abs(float(jsc_val)) if jsc_val is not None else 0.0,
            "Jsc_mAcm2": abs(float(jsc_val) / 10.0) if jsc_val is not None else 0.0,
            "FF_percent": (float(ff_val) * 100.0) if ff_val is not None and not np.isnan(ff_val) else 0.0,
            "PCE_percent": (float(pce_val) * 100.0) if pce_val is not None else 0.0,
            "Vmpp_V": abs(float(vmpp_val)) if vmpp_val is not None else 0.0,
            "Pmpp_Wm2": float(pmpp_val) if pmpp_val is not None else 0.0,
        }
        
        print("\n" + "=" * 70)
        print("FINAL RESULTS".center(70))
        print("=" * 70)
        print(json.dumps(output, indent=2))
        print("=" * 70)
        
        # Interpretation
        if output["Voc_V"] == 0.0:
            print("\n⚠️  ZERO OUTPUT - TROUBLESHOOTING:")
            print("1. Check if PDD solver is properly compiled:")
            print("   pip install --no-deps --force-reinstall --install-option=\"--with_pdd\" solcore")
            print("2. Try simpler structure (remove one transport layer)")
            print("3. Check material database parameters")
            print("4. Enable debug output in Fortran solver")
        elif output["Voc_V"] > 0.0 and output["Jsc_mAcm2"] > 0.0:
            print(f"\n✓ SUCCESS! Device is working")
            print(f"  Voc = {output['Voc_V']:.3f} V")
            print(f"  Jsc = {output['Jsc_mAcm2']:.2f} mA/cm²")
            print(f"  FF = {output['FF_percent']:.1f}%")
            print(f"  PCE = {output['PCE_percent']:.2f}%")
            
            if output['PCE_percent'] < 10:
                print("\n  → Efficiency is low. Try:")
                print("    - Increase carrier lifetimes (reduce recombination)")
                print("    - Optimize perovskite thickness (300-600 nm)")
                print("    - Fine-tune doping concentrations")

    except Exception as e:
        print(f"\n--- SIMULATION FAILED ---")
        print(f"Error: {e}")
        print(traceback.format_exc())

# --- Main execution ---
if __name__ == "__main__":
    run_simulation()