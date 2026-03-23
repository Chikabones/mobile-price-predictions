import joblib

model = joblib.load('model.pkl')

print("--- Mobile Price Predictor ---")
print("Enter phone specs:")

try:
    ram = float(input("RAM (GB): "))
    bat = float(input("Battery (mAh): "))
    mem = float(input("Internal Memory (GB): "))
    cpu = float(input("CPU Cores: "))
    cam = float(input("Rear Camera (MP): "))
    thick = float(input("Thickness (mm): "))

    user_data = [[ram, bat, mem, cpu, cam, thick]]
    prediction = model.predict(user_data)

    print(f"\nPredicted Price: ${round(prediction[0], 2)}")

except ValueError:
    print("\nError: Please enter numbers only!")