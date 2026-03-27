import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

np.random.seed(2024)
n = 300

complessita = np.random.uniform(1, 10, n)
carico      = np.random.uniform(1, 10, n)
distanza    = np.random.uniform(0.5, 12, n)
traffico    = np.random.uniform(1, 10, n)
pioggia     = np.random.choice([0, 1], n, p=[0.7, 0.3])   # [1]

tempo = (
    5 + complessita * 2.5 + carico * 1.5 +
    distanza * 3.5 * (1 + traffico * 0.08) +
    pioggia * 5 +
    np.random.normal(0, 2, n)
).clip(8, 100)

df = pd.DataFrame({'complessita': complessita, 'carico': carico,
                   'distanza': distanza, 'traffico': traffico,
                   'pioggia': pioggia, 'tempo_min': tempo})

print("SMARTFOOD — Dataset:", len(df), "ordini")
print(f"  Tempo medio: {tempo.mean():.1f} min  |  Min: {tempo.min():.0f}  |  Max: {tempo.max():.0f}")
print(df.head(4).round(1).to_string(index=False))

fattori   = ['complessita', 'carico', 'distanza', 'traffico']
etichette = ['Complessita', 'Carico cucina', 'Distanza', 'Traffico']

tempi_basso = [df[df[f] < df[f].median()]['tempo_min'].mean() for f in fattori]  # [2]
tempi_alto  = [df[df[f] >= df[f].median()]['tempo_min'].mean() for f in fattori]

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(fattori))
larghezza = 0.35

b1 = ax.bar(x - larghezza/2, tempi_basso, larghezza,  # [3]
            label='Valore basso', color='#2ecc71', edgecolor='black', linewidth=0.7)
b2 = ax.bar(x + larghezza/2, tempi_alto, larghezza,
            label='Valore alto', color='#e74c3c', edgecolor='black', linewidth=0.7)

ax.set_title('SMARTFOOD — Tempo Medio per Fattore (basso vs alto)', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(etichette)
ax.set_ylabel('Tempo consegna (min)')
ax.legend()
for b, v in zip(list(b1) + list(b2), tempi_basso + tempi_alto):
    ax.text(b.get_x() + b.get_width()/2, v + 0.3,
            f'{v:.0f}m', ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('smartfood_grafico.png', dpi=120, bbox_inches='tight')
plt.show()
print("Grafico salvato: smartfood_grafico.png")

X = df[['complessita', 'carico', 'distanza', 'traffico', 'pioggia']]
y = df['tempo_min']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modello = RandomForestRegressor(n_estimators=50, random_state=42)
modello.fit(X_train, y_train)
y_pred = modello.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)                               # [4]
print(f"\nErrore medio (MAE): {mae:.1f} minuti")
print(f"R2: {r2:.2f}  ({r2*100:.0f}% varianza spiegata)")

fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter( y_test, y_pred, alpha=0.4, s=25, color='#e67e22' )  # [5]
ax.plot([8, 100], [8, 100], 'r--', linewidth=1.5, label='Predizione perfetta')
ax.set_xlabel('Tempo Reale (min)'); ax.set_ylabel('Tempo Predetto (min)')
ax.set_title('SMARTFOOD — Reale vs Predetto', fontweight='bold')
ax.legend()
ax.text(10, 90, f'MAE={mae:.1f}min\nR2={r2:.2f}', fontsize=10,
        color='darkblue', fontweight='bold')
plt.tight_layout()
plt.savefig('smartfood_risultati.png', dpi=120, bbox_inches='tight')
plt.show()
print("Risultati salvati: smartfood_risultati.png")

print("\nSIMULATORE:")
nuovi = pd.DataFrame({'complessita': [8,   2,   6],
                       'carico':     [9,   2,   5],
                       'distanza':   [9.0, 1.5, 5.0],
                       'traffico':   [8,   2,   5],
                       'pioggia':    [1,   0,   0]})
ristoranti = ['Sushi (pioggia+traffico)', 'Burger Express (sereno)', 'Pizza Roma']
pred = modello.predict(nuovi)
for rist, t in zip(ristoranti, pred):
    stato = 'VELOCE' if t <= 25 else ('NORMALE' if t <= 45 else 'LENTO')  # [6]
    print(f"  {rist:30s}  →  {t:.0f} min  ({stato})")

    