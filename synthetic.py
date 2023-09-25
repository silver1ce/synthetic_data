import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Lataa todelliset tiedot CSV-tiedostosta (korvaa 'real_data.csv' omalla tiedostosi nimellä)
real_data = pd.read_csv('synthetic_grid_data.csv')

# Jaa tiedot ominaisuuksiin (X) ja kohdemuuttujaan (y)
X = real_data.drop('Node_1_Power_Flow', axis=1)  # Korvaa 'kohde_sarake' todellisen kohdesarakkeen nimellä
y = real_data['Node_1_Power_Flow']

# Luo synteettinen datasetti samalla määrällä näytteitä ja ominaisuuksia
synthetic_data, synthetic_labels = make_classification(
    n_samples=len(real_data),
    n_features=X.shape[1],
    n_informative=int(X.shape[1] * 0.8),  # Säädä informatiivisuutta tarpeen mukaan
    random_state=42,
)

# Jaa synteettiset tiedot koulutus- ja testisetteihin
X_train, X_test, y_train, y_test = train_test_split(synthetic_data, synthetic_labels, test_size=0.2, random_state=42)

# Kouluta luokittelija synteettisillä tiedoilla
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Arvioi luokittelijan tarkkuus todellisilla tiedoilla nähdäksesi, kuinka hyvin synteettiset tiedot suoriutuvat
accuracy = clf.score(X_test, y_test)
print(f"Accuracy on real data: {accuracy:.2f}")




# Arvioi luokittelija todellisilla tiedoilla
real_data_predictions = clf.predict(X_test)
real_data_cm = confusion_matrix(y_test, real_data_predictions)

# Arvioi luokittelija synteettisillä tiedoilla
synthetic_data_predictions = clf.predict(X_test)
synthetic_data_cm = confusion_matrix(y_test, synthetic_data_predictions)

# Luo yksi kuvaaja alikuvilla
plt.figure(figsize=(18, 6))

# Alikuva 1: Sekoitusmatriisi - Todelliset tiedot
plt.subplot(2, 3, 1)
sns.heatmap(real_data_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Real Data')

# Alikuva 2: Sekoitusmatriisi - Synteettiset tiedot
plt.subplot(2, 3, 2)
sns.heatmap(synthetic_data_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Synthetic Data')

# Alikuva 3: Pylväskaavio F1-pistemäärästä - Todelliset tiedot
plt.subplot(2, 3, 3)
real_data_report = classification_report(y_test, real_data_predictions, output_dict=True)
sns.barplot(x=['Class 0', 'Class 1'], y=[real_data_report['0']['f1-score'], real_data_report['1']['f1-score']])
plt.title('F1-Score - Real Data')
plt.ylabel('F1-Score')
plt.xlabel('Class')

# Alikuva 4: Pylväskaavio F1-pistemäärästä - Synteettiset tiedot
plt.subplot(2, 3, 4)
synthetic_data_report = classification_report(y_test, synthetic_data_predictions, output_dict=True)
sns.barplot(x=['Class 0', 'Class 1'], y=[synthetic_data_report['0']['f1-score'], synthetic_data_report['1']['f1-score']])
plt.title('F1-Score - Synthetic Data')
plt.ylabel('F1-Score')
plt.xlabel('Class')

# Alikuva 5: Hajontakaavio - Todelliset tiedot
plt.subplot(2, 3, 5)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot - Real Data')

# Alikuva 6: Hajontakaavio - Synteettiset tiedot
plt.subplot(2, 3, 6)
plt.scatter(X_test[:, 0], X_test[:, 1], c=synthetic_data_predictions, cmap='coolwarm')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot - Synthetic Data')

plt.tight_layout()
plt.savefig('visualized_report.png', bbox_inches='tight')
plt.show()

