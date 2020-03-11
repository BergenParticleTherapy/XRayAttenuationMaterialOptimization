import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, math
from scipy.optimize import minimize

def error_func(x, *args):
    mixture = {}
    xNorm = np.sum(x)
    for row in densities.iterrows():
        mixture[row[1][0]] = x[row[0]] / xNorm

    density = np.sum([ densities[densities["Element"] == k]["Density [g/cm3]"].values[0]*v for k,v in mixture.items() ])
    
    final_attenuation = pd.DataFrame()
    final_attenuation["Energy [keV]"] = energies
    final_attenuation["Linear attenuation [1/cm]"] = np.zeros(199)

    for k,v in mixture.items():
        idx = elements["Element"] == k
        if np.sum(idx) == 0: continue
        final_attenuation["Linear attenuation [1/cm]"] = final_attenuation["Linear attenuation [1/cm]"].add(elements["Linear attenuation [1/cm]"][idx] * v)

    final_attenuation = final_attenuation.merge(spectrum["Energy [keV]"], how='outer').sort_values("Energy [keV]").interpolate()
    attenuated_spectrum = pd.DataFrame()
    attenuated_spectrum["Energy [keV]"] = spectrum["Energy [keV]"]

    idx = 0
    attenuated_spectrum[f"Intensity"] = np.zeros(len(spectrum["Energy [keV]"]))
    for row in spectrum.iterrows():
        energy = row[1][0]
        intensity = row[1][1]
        attenuation = final_attenuation["Linear attenuation [1/cm]"][final_attenuation["Energy [keV]"] == energy].values[0]
        new_intensity = intensity * np.exp(-attenuation* 0.1)
        attenuated_spectrum[f"Intensity"][idx] = new_intensity
        idx += 1

    attenuated_spectrum.interpolate(inplace=True)
    tsum = attenuated_spectrum[f"Intensity"].sum()
    total_attenuation = tsum / orig_sum
    mass_attenuation = -math.log(total_attenuation)/(density*0.1)
    print(f"Spectrum-convoluted 1 mm mass attenuation of current test material is {mass_attenuation:.4f} cm2/g")

    # The line below defines which value to minimize
    return 1/mass_attenuation

densities = pd.read_csv("data/densities.txt", sep=";")
spectrum = pd.read_csv("data/scatter110kVp_AL.txt", sep=" ", header=None, names=["Energy [keV]", "Intensity"], index_col=False)
spectrum["Energy [keV]"] = 1000 * spectrum["Energy [keV]"]
orig_sum = spectrum["Intensity"].sum()

elements = pd.DataFrame()
for root, dirs, files in os.walk("data/Elements"):
    for file in files:
        df = pd.read_csv(f"data/Elements/{file}", skiprows=3,
                         names=["Energy [keV]", "Mass attenuation [cm2/g]", "dummy"],
                         header=None, sep=" ", index_col=False)
        df["Energy [keV]"] = df["Energy [keV]"] * 1000
        idx = densities["Element"] == file[:-4]
        if idx.sum() == 0:
            continue
        
        density = densities["Density [g/cm3]"][idx].values[0]
        df["Linear attenuation [1/cm]"] = df["Mass attenuation [cm2/g]"] * density
        df["Element"] = file[:-4]
        df.drop(columns="dummy")
        elements = elements.append(df)

energies = pd.Series(elements["Energy [keV]"][elements["Element"] == "Aluminum"])

n = len(densities)
bounds = ((0, 1), ) * n
x0 = np.ones(n) / n

res = minimize(error_func, x0, bounds=bounds)

x = res.x
mixture = {}
xNorm = np.sum(x)
for row in densities.iterrows():
    if x[row[0]]: mixture[row[1][0]] = x[row[0]] / xNorm

final_attenuation = pd.DataFrame()
final_attenuation["Energy [keV]"] = energies
final_attenuation["Linear attenuation [1/cm]"] = np.zeros(199)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5))

for k,v in mixture.items():
    idx = elements["Element"] == k
    final_attenuation["Energy [keV]"] = energies
    final_attenuation["Linear attenuation [1/cm]"] += elements["Linear attenuation [1/cm]"][idx] * v
    ax1.plot(elements["Energy [keV]"][idx], elements["Linear attenuation [1/cm]"][idx]*v, "--", label=f"{v*100:.1f}% {k}")

final_attenuation = final_attenuation.merge(spectrum["Energy [keV]"], how='outer').sort_values("Energy [keV]").interpolate()

ax1.plot(final_attenuation["Energy [keV]"], final_attenuation["Linear attenuation [1/cm]"], "-", label="Composite")
ax1.set_title("Material composite")
ax1.semilogy(True)
ax1.legend()
ax1.set_xlabel("Linear attenuation [1/cm]")
ax1.set_ylabel("Energy [keV]")
ax1.set_xlim([5, 115])

attenuated_spectrum = pd.DataFrame()
attenuated_spectrum["Energy [keV]"] = spectrum["Energy [keV]"]

density = np.sum([ densities[densities["Element"] == k]["Density [g/cm3]"].values[0]*v for k,v in mixture.items() ])
print(f"Density = {density}")
ax2.plot(spectrum["Energy [keV]"], spectrum["Intensity"], "-", label="Original spectrum")

orig_sum = spectrum["Intensity"].sum()
for d in [0.5, 1]: # linear thickness of material
    idx = 0
    dCM = d/10
    attenuated_spectrum[f"Intensity d={d} mm"] = np.zeros(len(spectrum["Energy [keV]"]))
    for row in spectrum.iterrows():
        energy = row[1][0]
        intensity = row[1][1]
        attenuation = final_attenuation["Linear attenuation [1/cm]"][final_attenuation["Energy [keV]"] == energy].values[0]
        new_intensity = intensity * np.exp(-attenuation * dCM)
        attenuated_spectrum[f"Intensity d={d} mm"][idx] = new_intensity
        idx += 1

    attenuated_spectrum.interpolate(inplace=True)
    ax2.plot(attenuated_spectrum["Energy [keV]"], attenuated_spectrum[f"Intensity d={d} mm"], "-", label=f"{d} mm material")

    total_attenuation = attenuated_spectrum[f"Intensity d={d} mm"].sum() / orig_sum
    print(f"Total attenuation for d={d} mm is {total_attenuation*100:.2e} %.")
    print(f"Mass attenuation µ/rho is {-math.log(total_attenuation)/(dCM*density):.2e} cm2/g")
    print(f"Linear attenuation µ is {-math.log(total_attenuation)/dCM:.2e} 1/cm\n")
    
ax2.legend()
ax2.set_title("Attenuated spectrum")
ax2.set_xlim([5, 115])
ax2.set_xlabel("Spectrum intensity")
ax2.set_ylabel("Energy [keV]")
plt.tight_layout(2)
plt.show()
