import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# --- Illuminant D65 (nm, SPD relative) ---
d65 = np.loadtxt("CIE_std_illum_D65.csv", delimiter=",")
d65_wl, d65_val = d65[:, 0], d65[:, 1]


def _convert(text):
    """Texte -> bool / float / str."""
    t = text.strip()
    if t in ("true", "false"):
        return t == "true"
    try:
        return float(t)
    except ValueError:
        return t


def _parse_node(node):
    children = list(node)
    if not children:
        return _convert(node.text or "")

    out = {child.tag: _parse_node(child) for child in children}

    # Tableau de valeurs : remis en forme avec num_rows / num_columns
    if "values" in out and isinstance(out["values"], str):
        flat = np.array(out["values"].split(), dtype=float)
        if "num_rows" in out and "num_columns" in out:
            flat = flat.reshape(int(out["num_rows"]), int(out["num_columns"]))
        out["values"] = flat
    elif "values" in out and isinstance(out["values"], float):   # une seule valeur
        out["values"] = np.array([out["values"]])
    return out


def parse_igm(path):
    """Retourne un dict imbriqué : data['scenedata']['material'][<balise>] ..."""
    root = ET.parse(path).getroot()
    return {root.tag: _parse_node(root)}

# --- Bruton (repris de SpectraToRgb.py) : longueur d'onde (nm) -> RGB 0..1 ---
def bruton(w):
    if w < 380 or w > 780: R = G = B = 0.0
    elif w < 440: R, G, B = -(w - 440) / 60, 0.0, 1.0
    elif w < 490: R, G, B = 0.0, (w - 440) / 50, 1.0
    elif w < 510: R, G, B = 0.0, 1.0, -(w - 510) / 20
    elif w < 580: R, G, B = (w - 510) / 70, 1.0, 0.0
    elif w < 645: R, G, B = 1.0, -(w - 645) / 65, 0.0
    else:         R, G, B = 1.0, 0.0, 0.0
    if w < 380 or w > 780: f = 0.0
    elif w < 420: f = 0.3 + 0.7 * (w - 380) / 40
    elif w < 701: f = 1.0
    else:         f = 0.3 + 0.7 * (780 - w) / 80
    return np.array([R, G, B]) * f


# data = parse_igm("SGG-COOL-LITE-ST-Bright-Silver-m.igm")
# mat  = data["scenedata"]["material"]
# spec = mat["specular"]
# transmission = spec["tabulated_transmission_coefficient"]
# start_transmission= transmission["start_wavelength"]
# end_transmission = transmission["end_wavelength"]
# vals  = transmission["values"]
# wl = np.linspace(start_transmission, end_transmission, vals.shape[0]) 

data = parse_igm("sgg-planiclear-medium.igm")
absorption  = data["scenedata"]["medium"]["basic"]["absorption_coefficient_spectrum"]["regular_tabulated"]
start_absorption  = absorption["start_wavelength"]
end_absorption  = absorption["end_wavelength"]

alpha  = absorption["values"]
wl_nm = np.linspace(start_absorption, end_absorption, alpha.shape[0]) * 1e9   

illum = np.interp(wl_nm, d65_wl, d65_val)                      # D65 sur nos 81 λ
base  = np.array([bruton(w) for w in wl_nm])                   # (81, 3)
white = (base * illum[:, None]).sum(axis=0)                    # albédo = 1 partout

def absorption_colors(d_array, gamma=0.8):
    """Couleurs RGB (N, 3) pour un tableau d'épaisseurs d_array (m)."""
    T   = np.exp(-alpha[None, :] * d_array[:, None])       # (N, 81)
    mat = (T * illum[None, :]) @ base                      # (N, 3)
    return np.clip(mat / white, 0, 1) ** gamma

def is_black(d, gamma=0.8):
    rgb = absorption_colors(np.array([d]), gamma)[0]
    return np.all(np.round(rgb * 255) < 20 )

# --- 1) Trouver d_black : bornes puis dichotomie ---
low, high = 0.0, 1.0
while not is_black(high):          # on double jusqu'à trouver du noir
    low, high = high, high * 2
for _ in range(40):              # dichotomie (la luminosité décroît avec d)
    mid = 0.5 * (low + high)
    if is_black(mid): high = mid
    else:             low = mid
d_black = high
print(f"Noir (0,0,0) à partir de {d_black:.3f} m")

# --- 2) Bande continue de 0 à d_black ---
N = 1024
d = np.linspace(0, d_black, N)
strip = absorption_colors(d)

fig, ax = plt.subplots(figsize=(10, 2.2))
ax.imshow(strip[None, :, :], aspect="auto", extent=[0, d_black, 0, 1])
ax.set_yticks([])

# --- 3) Échelle graduée sur [0, d_black] ---
ticks = np.linspace(0, d_black, 15)
ax.set_xticks(ticks)
ax.set_xticklabels([f"{t:.2f} m" for t in ticks])
ax.set_xlabel(f"Épaisseur de verre (noir à {d_black:.2f} m)")
plt.tight_layout()
plt.show()

# illum = np.interp(transmission, d65_wl, d65_val)          # D65 rééchantillonné sur nos 15 λ
# base  = np.array([bruton(w) for w in wl])
# white = (base * illum[:, None]).sum(axis=0)              # blanc : albédo = 1 partout
# mat   = (base * (vals * illum)[:, None]).sum(axis=0) 

# gamma = 0.8

# rgb = np.clip(mat / white, 0, 1) ** gamma 

# # --- Image ---
# plt.imshow([[rgb]])
# plt.axis("off")
# plt.title("#%02x%02x%02x" % tuple(int(round(c * 255)) for c in rgb))
# plt.show()