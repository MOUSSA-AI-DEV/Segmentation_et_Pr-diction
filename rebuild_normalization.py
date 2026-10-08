import json

with open('file.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Keep only cells 0-26 (the original cells before normalization blocks)
nb['cells'] = nb['cells'][:27]

# Now add clean normalization cells (single time)
new_cells = [
    # ---- MARKDOWN: Justification intro ----
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Préparation des variables RFM avant K-Means\n",
            "\n",
            "Avant d'appliquer l'algorithme **K-Means**, plusieurs transformations sont nécessaires.\n",
            "K-Means repose sur des **distances euclidiennes** entre les points. Si les variables ont\n",
            "des échelles très différentes ou des distributions très asymétriques, les variables à grande\n",
            "échelle domineront le calcul de distance et fausseront les résultats du clustering.\n",
            "\n",
            "Les étapes clés sont :\n",
            "1. **Vérification des outliers** – les valeurs extrêmes perturbent la distance euclidienne.\n",
            "2. **Transformation logarithmique** – pour réduire l'asymétrie (skewness) des distributions.\n",
            "3. **Standardisation (StandardScaler)** – pour mettre toutes les variables sur la même échelle (moyenne=0, écart-type=1)."
        ]
    },
    # ---- MARKDOWN: Step 1 ----
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Étape 1 : Vérification de la distribution et de l'asymétrie (Skewness)\n",
            "\n",
            "Avant toute transformation, on vérifie la **skewness** (asymétrie) de chaque variable.\n",
            "Une skewness élevée (> 1) indique une distribution très asymétrique, ce qui peut nuire à K-Means."
        ]
    },
    # ---- CODE: Skewness check ----
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "X = rfm[[\"Recency\", \"Frequency\", \"Monetary\"]].copy()\n",
            "\n",
            "print(\"=== Skewness AVANT transformation ===\")\n",
            "print(X.skew())\n",
            "\n",
            "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n",
            "for i, col in enumerate([\"Recency\", \"Frequency\", \"Monetary\"]):\n",
            "    axes[i].hist(X[col], bins=40, color='steelblue', edgecolor='white')\n",
            "    axes[i].set_title(f'Distribution de {col}\\nSkewness = {X[col].skew():.2f}')\n",
            "    axes[i].set_xlabel(col)\n",
            "    axes[i].set_ylabel('Nombre de clients')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    # ---- MARKDOWN: Step 2 ----
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Étape 2 : Transformation logarithmique\n",
            "\n",
            "**Pourquoi ?**\n",
            "- `Frequency` et `Monetary` ont souvent une distribution très asymétrique à droite.\n",
            "- La transformation `log1p` (= log(x+1)) compresse les grandes valeurs et rend la distribution plus symétrique.\n",
            "- On utilise `log1p` au lieu de `log` pour éviter les erreurs sur les valeurs égales à 0.\n",
            "\n",
            "**On applique `log1p` aux 3 variables** car toutes peuvent être asymétriques."
        ]
    },
    # ---- CODE: Log transform ----
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "X_log = X.copy()\n",
            "\n",
            "X_log[\"Recency\"]   = np.log1p(X[\"Recency\"])\n",
            "X_log[\"Frequency\"] = np.log1p(X[\"Frequency\"])\n",
            "X_log[\"Monetary\"]  = np.log1p(X[\"Monetary\"])\n",
            "\n",
            "print(\"=== Skewness APRES transformation log1p ===\")\n",
            "print(X_log.skew())\n",
            "\n",
            "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n",
            "for i, col in enumerate([\"Recency\", \"Frequency\", \"Monetary\"]):\n",
            "    axes[i].hist(X_log[col], bins=40, color='darkorange', edgecolor='white')\n",
            "    axes[i].set_title(f'Apres log1p : {col}\\nSkewness = {X_log[col].skew():.2f}')\n",
            "    axes[i].set_xlabel(col)\n",
            "    axes[i].set_ylabel('Nombre de clients')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    # ---- MARKDOWN: Step 3 ----
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Étape 3 : Standardisation avec StandardScaler\n",
            "\n",
            "**Pourquoi ?**\n",
            "- Apres la transformation log, les variables sont sur des echelles differentes.\n",
            "- **K-Means utilise la distance euclidienne**, donc une variable avec une grande echelle dominera toujours le clustering.\n",
            "- `StandardScaler` transforme chaque variable pour avoir **moyenne = 0** et **ecart-type = 1**.\n",
            "\n",
            "> **Pourquoi StandardScaler et non MinMaxScaler ?**\n",
            "> StandardScaler est prefere car il est moins sensible aux outliers restants."
        ]
    },
    # ---- CODE: StandardScaler ----
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.preprocessing import StandardScaler\n",
            "import pandas as pd\n",
            "\n",
            "scaler = StandardScaler()\n",
            "X_scaled = scaler.fit_transform(X_log)\n",
            "\n",
            "X_scaled = pd.DataFrame(X_scaled, columns=[\"Recency\", \"Frequency\", \"Monetary\"])\n",
            "\n",
            "print(\"=== Statistiques APRES standardisation ===\")\n",
            "print(X_scaled.describe().round(3))\n",
            "\n",
            "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n",
            "for i, col in enumerate([\"Recency\", \"Frequency\", \"Monetary\"]):\n",
            "    axes[i].hist(X_scaled[col], bins=40, color='seagreen', edgecolor='white')\n",
            "    axes[i].set_title(f'Apres StandardScaler : {col}\\nMoyenne={X_scaled[col].mean():.2f}, Std={X_scaled[col].std():.2f}')\n",
            "    axes[i].set_xlabel(col)\n",
            "    axes[i].set_ylabel('Nombre de clients')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    # ---- MARKDOWN: Summary ----
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Resume des transformations appliquees\n",
            "\n",
            "| Etape | Transformation | Justification |\n",
            "|-------|---------------|---------------|\n",
            "| 1 | Verification skewness | Identifier les variables asymetriques |\n",
            "| 2 | `log1p(x)` | Reduire l'asymetrie des distributions |\n",
            "| 3 | `StandardScaler` | Egaliser les echelles pour la distance euclidienne de K-Means |\n",
            "\n",
            "Le DataFrame `X_scaled` est maintenant pret pour **K-Means**."
        ]
    }
]

nb['cells'].extend(new_cells)

with open('file.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Done! Notebook now has {len(nb['cells'])} cells.")
