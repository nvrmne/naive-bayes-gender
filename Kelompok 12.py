import csv
import math
from statistics import mean, stdev

def load_dataset(filename):
    data = []
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # convert angka
            row["long_hair"] = int(row["long_hair"])
            row["nose_wide"] = int(row["nose_wide"])
            row["nose_long"] = int(row["nose_long"])
            row["lips_thin"] = int(row["lips_thin"])
            row["distance_nose_to_lip_long"] = int(row["distance_nose_to_lip_long"])
            row["forehead_width_cm"] = float(row["forehead_width_cm"])
            row["forehead_height_cm"] = float(row["forehead_height_cm"])
            data.append(row)
    return data

def split_data(data):
    n_train = int(0.8 * len(data))
    train = data[:n_train]
    test = data[n_train:]
    return train, test

def gaussian(x, mu, sigma):
    return (1 / (math.sqrt(2*math.pi) * sigma)) * math.exp(-((x - mu)**2) / (2 * sigma**2))

def train_model(train):
    # pisahkan male & female
    male = [d for d in train if d["gender"] == "Male"]
    female = [d for d in train if d["gender"] == "Female"]

    model = {}

    # prior
    model["prior_male"] = len(male) / len(train)
    model["prior_female"] = len(female) / len(train)

    # fitur biner → hitung frekuensi
    fitur_biner = ["long_hair","nose_wide","nose_long","lips_thin","distance_nose_to_lip_long"]
    for f in fitur_biner:
        model[f+"_1_male"] = sum(d[f] == 1 for d in male)
        model[f+"_0_male"] = sum(d[f] == 0 for d in male)

        model[f+"_1_female"] = sum(d[f] == 1 for d in female)
        model[f+"_0_female"] = sum(d[f] == 0 for d in female)

    # fitur kontinu → mean dan std
    fw_m = [d["forehead_width_cm"] for d in male]
    fw_f = [d["forehead_width_cm"] for d in female]
    fh_m = [d["forehead_height_cm"] for d in male]
    fh_f = [d["forehead_height_cm"] for d in female]

    model["fw_m_mean"] = mean(fw_m)
    model["fw_f_mean"] = mean(fw_f)
    model["fw_m_std"]  = stdev(fw_m)
    model["fw_f_std"]  = stdev(fw_f)

    model["fh_m_mean"] = mean(fh_m)
    model["fh_f_mean"] = mean(fh_f)
    model["fh_m_std"]  = stdev(fh_m)
    model["fh_f_std"]  = stdev(fh_f)

    return model

def predict(model, d):
    # log prior
    log_m = math.log(model["prior_male"])
    log_f = math.log(model["prior_female"])

    # fitur biner
    fitur = ["long_hair","nose_wide","nose_long","lips_thin","distance_nose_to_lip_long"]

    for f in fitur:
        if d[f] == 1:
            log_m += math.log(model[f+"_1_male"])
            log_f += math.log(model[f+"_1_female"])
        else:
            log_m += math.log(model[f+"_0_male"])
            log_f += math.log(model[f+"_0_female"])

    # gaussian
    log_m += math.log(gaussian(d["forehead_width_cm"], model["fw_m_mean"], model["fw_m_std"]))
    log_f += math.log(gaussian(d["forehead_width_cm"], model["fw_f_mean"], model["fw_f_std"]))

    log_m += math.log(gaussian(d["forehead_height_cm"], model["fh_m_mean"], model["fh_m_std"]))
    log_f += math.log(gaussian(d["forehead_height_cm"], model["fh_f_mean"], model["fh_f_std"]))

    return "Male" if log_m > log_f else "Female"

def evaluate(model, test):
    benar = 0
    for d in test:
        pred = predict(model, d)
        if pred == d["gender"]:
            benar += 1
    return benar / len(test)

def menu_input_user():
    print("\n=== INPUT DATA UJI ===")
    d = {}
    d["long_hair"] = int(input("long_hair (0/1): "))
    d["forehead_width_cm"] = float(input("forehead_width_cm: "))
    d["forehead_height_cm"] = float(input("forehead_height_cm: "))
    d["nose_wide"] = int(input("nose_wide (0/1): "))
    d["nose_long"] = int(input("nose_long (0/1): "))
    d["lips_thin"] = int(input("lips_thin (0/1): "))
    d["distance_nose_to_lip_long"] = int(input("distance_nose_to_lip_long (0/1): "))
    return d

def main():
    data = load_dataset("gender_classification_v7.csv")
    train, test = split_data(data)
    model = train_model(train)

    print("\n=== PERHITUNGAN PROBABILITAS SETIAP KELAS ===")
    print(f"P(Male)   = {model['prior_male']:.6f}")
    print(f"P(Female) = {model['prior_female']:.6f}\n")

    fitur_biner = ["long_hair","nose_wide","nose_long","lips_thin","distance_nose_to_lip_long"]

    for f in fitur_biner:
        print(f"{f}:")
        print(f"  P({f}=1 | Male)   = {model[f+'_1_male']}")
        print(f"  P({f}=0 | Male)   = {model[f+'_0_male']}")
        print(f"  P({f}=1 | Female) = {model[f+'_1_female']}")
        print(f"  P({f}=0 | Female) = {model[f+'_0_female']}")
        print("")

    print("Mean & Std (Gaussian):")
    print(f"  Mean FW Male   = {model['fw_m_mean']:.5f}, Std = {model['fw_m_std']:.5f}")
    print(f"  Mean FW Female = {model['fw_f_mean']:.5f}, Std = {model['fw_f_std']:.5f}")
    print(f"  Mean FH Male   = {model['fh_m_mean']:.5f}, Std = {model['fh_m_std']:.5f}")
    print(f"  Mean FH Female = {model['fh_f_mean']:.5f}, Std = {model['fh_f_std']:.5f}")

    
    # user test
    d = menu_input_user()
    hasil = predict(model, d)
    print("\nPrediksi hasil:", hasil)

    # akurasi
    acc = evaluate(model, test)
    print(f"\nAkurasi model pada data testing: {acc*100:.2f}%")
    
input("\nTekan ENTER untuk keluar...")

main()
