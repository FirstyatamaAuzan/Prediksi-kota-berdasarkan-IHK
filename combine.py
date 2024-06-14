import csv
import math
import os

# Definisikan jalur ke file CSV
filename = 'tubes\datanew.csv'

# Membaca data dari file CSV
data = []
with open(filename) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        data.append(row)

# Mengubah data string menjadi float untuk kolom yang sesuai
for i in range(1, len(data)):
    for j in range(1, 4):
        data[i][j] = float(data[i][j])

# Definisikan entropi
def entropy(data):
    total = len(data)
    counts = {}
    for row in data:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    
    ent = 0
    for label in counts:
        p = counts[label] / total
        ent -= p * math.log2(p)
    return ent

# Memisahkan data berdasarkan nilai atribut tertentu
def split_data(data, attribute, value):
    true_rows = [row for row in data if row[attribute] >= value]
    false_rows = [row for row in data if row[attribute] < value]
    return true_rows, false_rows

# Menghitung informasi keuntungan dari split data
def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * entropy(left) - (1 - p) * entropy(right)

# Menentukan split terbaik
def find_best_split(data):
    best_gain = 0
    best_attribute = None
    best_value = None
    current_uncertainty = entropy(data)
    n_features = len(data[0]) - 2  # Jumlah kolom tanpa kolom Kota dan Decision
    
    for col in range(1, n_features + 1):  # Mulai dari 1 karena kolom pertama adalah Kota
        values = set([row[col] for row in data])
        for val in values:
            true_rows, false_rows = split_data(data, col, val)
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            gain = info_gain(true_rows, false_rows, current_uncertainty)
            if gain >= best_gain:
                best_gain, best_attribute, best_value = gain, col, val
    
    return best_gain, best_attribute, best_value

# Definisikan node daun
class Leaf:
    def __init__(self, data):
        counts = {}
        for row in data:
            label = row[-1]
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        self.predictions = counts

# Definisikan node keputusan
class DecisionNode:
    def __init__(self, attribute, value, true_branch, false_branch):
        self.attribute = attribute
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch

# Membangun decision tree
def build_tree(data):
    gain, attribute, value = find_best_split(data)
    if gain == 0:
        return Leaf(data)
    true_rows, false_rows = split_data(data, attribute, value)
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)
    return DecisionNode(attribute, value, true_branch, false_branch)

# Membuat prediksi menggunakan decision tree
def classify(row, node):
    if isinstance(node, Leaf):
        return max(node.predictions, key=node.predictions.get)
    
    if row[node.attribute] >= node.value:
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

# Membangun decision tree dari data
tree = build_tree(data[1:])

# Fungsi untuk membuat prediksi untuk input baru
def predict_new_data():
    while True:
        nama_kota = input("Nama Kota (atau ketik 'exit' untuk keluar): ")
        if nama_kota.lower() == 'exit':
            print("Terima kasih! Sampai jumpa, Babaiii.")
            break
        
        transportasi_pribadi = float(input("Transportasi Pribadi: "))
        jasa_angkutan_penumpang = float(input("Jasa Angkutan Penumpang: "))
        jasa_pengiriman_barang = float(input("Jasa Pengiriman Barang: "))
        
        new_row = [nama_kota, transportasi_pribadi, jasa_angkutan_penumpang, jasa_pengiriman_barang]
        decision = classify(new_row, tree)
        print(f"Prediksi untuk Kota {nama_kota}: {decision}")

# Data uji
test_data = [
    ["SINGARAJA", 123.8333333, 141.1666667, 100, "Ramai"],
    ["KOTA DENPASAR", 121.9166667, 118, 118.1666667, "Tidak Ramai"],
    ["KOTA MATARAM", 123.1666667, 149.9166667, 114.3333333, "Tidak Ramai"],
    ["KOTA BIMA", 122.8333333, 205.5833333, 115.8333333, "Ramai"],
    ["WAINGAPU", 131, 114.0833333, 99, "Tidak Ramai"],
    ["MAUMERE", 125.5, 158.8333333, 136.5, "Ramai"],
    ["KOTA KUPANG", 122.5833333, 161.5833333, 124.3333333, "Ramai"],
    ["SINTANG", 126.4166667, 182.8333333, 90, "Ramai"],
    ["KOTA PONTIANAK", 127.3333333, 122.8333333, 124.9166667, "Ramai"],
    ["KOTA SINGKAWANG", 124.75, 127, 109.8333333, "Tidak Ramai"]]

# Membuat prediksi untuk data uji
predictions = [classify(row[:-1], tree) for row in test_data]

# Menghitung TP, TN, FP, FN
TP = sum(1 for i, row in enumerate(test_data) if row[-1] == "Ramai" and predictions[i] == "Ramai")
TN = sum(1 for i, row in enumerate(test_data) if row[-1] == "Tidak Ramai" and predictions[i] == "Tidak Ramai")
FP = sum(1 for i, row in enumerate(test_data) if row[-1] == "Tidak Ramai" and predictions[i] == "Ramai")
FN = sum(1 for i, row in enumerate(test_data) if row[-1] == "Ramai" and predictions[i] == "Tidak Ramai")

# Menghitung metrik evaluasi
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
f1_score = 2 * (recall * precision) / (recall + precision) if (recall + precision) > 0 else 0

# Menampilkan hasil
print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
print(f"Akurasi: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Specificity: {specificity}")
print(f"F1 Score: {f1_score}")

# Meminta input untuk data baru dan membuat prediksi
predict_new_data()
