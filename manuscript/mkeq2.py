#!/usr/bin/env python3
"""Render equations as narrow, column-width images (multi-part ones stacked)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import os

plt.rcParams['mathtext.fontset'] = 'cm'

# each equation is a list of lines; multi-part equations stack
EQ = {
 1: [r"$\kappa = \dfrac{p_o - p_e}{1 - p_e},\qquad p_o = \dfrac{1}{N}\sum_{i=1}^{N}\mathbb{1}\left[y_i^{(1)} = y_i^{(2)}\right],$",
     r"$p_e = \sum_{k \in \{0,1\}} P\left(y^{(1)}\!=\!k\right)P\left(y^{(2)}\!=\!k\right)$"],
 2: [r"$w_{t,d} = \mathrm{TF}(t,d)\cdot\left[\log\left(\dfrac{1+N}{1+\mathrm{DF}(t)}\right)+1\right],$",
     r"$\hat{x}_d = x_d\,/\,\|x_d\|_2$"],
 3: [r"$\min_{w,b}\ \ \dfrac{1}{2}\|w\|_2^2 + C\sum_{i=1}^{n}\max\left(0,\ 1-\tilde{y}_i\left(w^{\top}\hat{x}_i + b\right)\right)$"],
 4: [r"$\hat{y} = \arg\max_{c}\ \log\hat{P}(c) + \sum_{t=1}^{|V|}\hat{x}_t\log\hat{P}(t\mid c),$",
     r"$\hat{P}(t\mid c) = \dfrac{\alpha + \sum_{d \in c}\hat{x}_{t,d}}{\alpha|V| + \sum_{t'}\sum_{d \in c}\hat{x}_{t',d}}$"],
 5: [r"$\mathrm{score}(t) = \log\dfrac{\left(1+\sum_{d \in c_1}\mathbb{1}[t \in d]\right)/n_1}{\left(1+\sum_{d \in c_0}\mathbb{1}[t \in d]\right)/n_0},$",
     r"$\hat{y} = \mathbb{1}\left[\exists\, t \in L_k : t \in d\right]$"],
 6: [r"$\mathcal{L} = -\dfrac{1}{n}\sum_{i=1}^{n}\sum_{c \in \{0,1\}} y_{i,c}\log p_{i,c}$"],
 7: [r"$\left\{F_1^{\mathrm{macro}}\left(\mathcal{B}^{(b)}\right)\right\}_{b=1}^{B},$",
     r"$\mathcal{B}^{(b)} \sim \mathrm{Multinomial}\left(N,\ \frac{1}{N}\mathbf{1}\right)$"],
 8: [r"$\chi^{2} = \dfrac{\left(|n_{01}-n_{10}| - 1\right)^{2}}{n_{01}+n_{10}}$"],
 9: [r"$J^{*}(d) = \max_{d' \in D_{\mathrm{train}}}\dfrac{|S(d) \cap S(d')|}{|S(d) \cup S(d')|}$"],
}

os.makedirs('eq2', exist_ok=True)

def render_line(tex, path, fs=13):
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, tex, fontsize=fs)
    fig.savefig(path, dpi=400, bbox_inches='tight', pad_inches=0.06,
                facecolor='white', transparent=False)
    plt.close(fig)

for n, lines in EQ.items():
    parts = []
    for j, ln in enumerate(lines):
        p = f'eq2/_t{n}_{j}.png'
        render_line(ln, p)
        parts.append(Image.open(p))
    if len(parts) == 1:
        img = parts[0]
    else:
        w = max(p.size[0] for p in parts)
        gap = 14
        h = sum(p.size[1] for p in parts) + gap * (len(parts) - 1)
        img = Image.new('RGB', (w, h), 'white')
        y = 0
        for p in parts:
            img.paste(p, ((w - p.size[0]) // 2, y))   # centre each line
            y += p.size[1] + gap
    img.save(f'eq2/eq{n}.png')
    for j in range(len(lines)):
        os.remove(f'eq2/_t{n}_{j}.png')
    W, H = img.size
    print(f"eq{n}: {W}x{H}  aspect {H/W:.3f}  -> at 2.9in wide = {2.9*H/W:.2f}in tall")
