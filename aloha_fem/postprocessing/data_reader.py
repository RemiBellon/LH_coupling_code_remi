import h5py
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class WaveguidePort:
    index: int
    type: str          # "active" or "passive"
    z_start: float
    z_end: float
    length: float
    phase_deg: float
    power_reflectivity: float

@dataclass
class PhysicsMetadata:
    freq_LH: float
    n_para_req: float

@dataclass
class ElectricalFields:
    z_coords: np.ndarray
    Ex: np.ndarray
    Ey: np.ndarray
    Ez: np.ndarray
    E_norm: np.ndarray

@dataclass
class PowerSpectrum:
    n_para: np.ndarray
    dP_dn_para: np.ndarray

class SimulationData:
    def __init__(self, h5_filepath: str):
        """
        Loads the timestamped HDF5 database and maps it to dot-notation attributes.
        """
        self.filepath = h5_filepath
        with h5py.File(h5_filepath, "r") as f:
            # Metadata
            self.meta = PhysicsMetadata(
                freq_LH=f["Metadata"].attrs["freq_LH"],
                n_para_req=f["Metadata"].attrs["n_para_req"]
            )
            
            # 1D Spectrum
            self.spectrum = PowerSpectrum(
                n_para=f["Power_Spectrum/n_para"][:],
                dP_dn_para=f["Power_Spectrum/dP_dn_para"][:]
            )
            
            # 1D Tangential Fields (Rebuilding complex arrays)
            grp = f["Tangential_Fields"]
            ex = grp["Ex_real"][:] + 1j * grp["Ex_imag"][:]
            ey = grp["Ey_real"][:] + 1j * grp["Ey_imag"][:]
            ez = grp["Ez_real"][:] + 1j * grp["Ez_imag"][:]
            e_norm = np.sqrt(np.abs(ex)**2 + np.abs(ey)**2 + np.abs(ez)**2)
            
            self.fields = ElectricalFields(
                z_coords=grp["z_coords"][:],
                Ex=ex, Ey=ey, Ez=ez, E_norm=e_norm
            )

            if "Map_2D" in f:
                grp2 = f["Map_2D"]
                self.map_2d = {
                    "X": grp2["X"][:],
                    "Z": grp2["Z"][:],
                    "Ex": grp2["Ex_real"][:] + 1j * grp2["Ex_imag"][:],
                    "Ey": grp2["Ey_real"][:] + 1j * grp2["Ey_imag"][:],
                    "Ez": grp2["Ez_real"][:] + 1j * grp2["Ez_imag"][:]
                }
            else:
                self.map_2d = None

        self.ports: List[WaveguidePort] = []
        with h5py.File(h5_filepath, "r") as f:
            for key, val in f.attrs.items():
                                        if isinstance(val, (float, np.floating)):
                                            print(f' {key}: {val:.2e}')
                                        else:
                                            print(f' {key}: {val}')
            if "S_Parameters" in f:
                s_grp = f["S_Parameters"]
                for port_key in sorted(s_grp.keys(), key=lambda x: int(x.split('_')[1])):
                    p_data = s_grp[port_key]
                    self.ports.append(WaveguidePort(
                        index=int(port_key.split('_')[1]),
                        type=p_data.attrs.get("type", "active"),
                        z_start=p_data.attrs.get("z_start", 0.0),
                        z_end=p_data.attrs.get("z_end", 0.0),
                        length=p_data.attrs.get("length", 0.0),
                        phase_deg=p_data.attrs.get("Phase_deg", 0.0),
                        power_reflectivity=p_data.attrs.get("Power_Reflectivity", 0.0)
                    ))
            