# 🧬 GA-Based Feature Selection for Phishing Email Detection  

Phishing emails are one of the most dangerous cyber threats today.  
This project applies a **Genetic Algorithm (GA)** to automatically select the most important **textual** and **metadata-based** features from emails, improving phishing detection while keeping the system lightweight, explainable, and fully offline.  

---

## 🚀 Why This Project?  
🔹 Most phishing detection tools are web-based — this one runs **locally**.  
🔹 GA helps select the **most relevant features** from email text & headers.  
🔹 Improves classification accuracy while reducing noise and overfitting.  


## ⚡ Features  
✔️ Extracts **text features** (word frequency, exclamation marks, suspicious URLs).  
✔️ Extracts **header features** (SPF/DKIM failures, sender mismatch, return-path anomalies).  
✔️ Applies **Genetic Algorithm (GA)** for feature selection.  
✔️ Trains & evaluates using **Support Vector Machine (SVM)** with cross-validation.  
✔️ Saves outputs:  
   - **Selected features** list  
   - **Fitness curve** plot  
   - **Metrics report** (Accuracy, Precision, Recall, F1)  

---
## 🛡️ Use Cases

🔹 Academic projects in Cybersecurity & AI/ML
🔹 Research in feature selection optimization
🔹 Proof-of-concept for phishing/spam classification
🔹 Teaching demo for GA applied in security

---

## 🔮 Future Work

📌 Try on real phishing datasets (e.g., Enron, SpamAssassin).

📌 Add multi-objective GA (maximize F1, minimize features).

📌 Experiment with Random Forest, XGBoost, or Deep Learning classifiers.

---

## 🤝 Contributing

Pull requests are welcome!

Fork the repo

Create a new branch (feature-xyz)

Commit changes & open a PR 🚀

---

## 🛠️ Installation & Setup  

### 1️⃣ Clone the Repo

[git clone https://github.com/your-username/ga-phishing-feature-selection.git
cd ga-phishing-feature-selection]

---

### 2️⃣ Install Requirements

pip install -r requirements.txt

---

### 3️⃣ Run GA on Sample Dataset

python ga_phishing.py --csv sample_emails.csv --pop 20 --gens 10


## 📊 Example Output

### ✅ GA Fitness Curve

### ✅ Selected Features

['num_urls', 'spf_fail', 'dkim_fail', 'has_ip_url', 'sender_suspicious_tld']


### ✅ Metrics Report

Baseline SVM F1: 0.71
GA-selected SVM F1: 0.83

---

## ⭐ Support

If this repo helped you, consider giving it a ⭐ — it motivates further improvements and makes the project more visible!

---
## 📢 Author  

**Atrima Bhattacharyya**  
- 🎓 MCA Student, UEM University of Engineering and Management  
- 💻 Cybersecurity & AI/ML Enthusiast  

