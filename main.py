import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Constants
P_max = 132  # Maximum power in kW
nu = 0.7  # Efficiency coefficient
n_max =25.3  # Maximum rpm ====== this value is a constant based on the reference but can be adjustable based on the descriptive analysis
n_min = 21 # Minimum rpm for data filtering === the min value can be selected based on 75% quartile of data based on descriptive table
x_axis_max = 26  # Extended x-axis for visualization

# Generate rpm values for the continuous curves
rpm_curve = np.linspace(0.1, n_max, 1000)  # Avoid division by zero

# Use known values for M_cont and M_max_Vg1
M_cont_value = 44  # Known continuous torque in kNm
M_max_Vg1 = 54  # Maximum torque for Vg1 in kNm

# Function to calculate M max Vg2
def M_max_Vg2(rpm):
    return np.minimum(M_max_Vg1, (P_max * 60 * nu) / (2 * np.pi * rpm))

# Calculate the intersection points
elbow_rpm_max = (P_max * 60 * nu) / (2 * np.pi * M_max_Vg1)
elbow_rpm_cont = (P_max * 60 * nu) / (2 * np.pi * M_cont_value)

# Read the CSV file
file_path = "/content/drive/MyDrive/SW/BN SW 27 700.csv"  # Update this path as needed
df = pd.read_csv(file_path, sep=';', decimal=',')

# Rename columns for clarity
df = df.rename(columns={
    'AzV.V13_SR_ArbDr_Z | DB    60.DBD    26': 'Working pressure [bar]',
    'AzV.V13_SR_Drehz_nach_Abgl_Z | DB    60.DBD    30': 'Revolution [rpm]'
})

# Clean numeric columns
df['Revolution [rpm]'] = pd.to_numeric(df['Revolution [rpm]'], errors='coerce')
df['Working pressure [bar]'] = pd.to_numeric(df['Working pressure [bar]'], errors='coerce')

# Remove rows with NaN values
df = df.dropna(subset=['Revolution [rpm]', 'Working pressure [bar]'])

# Filter data points between n_min and n_max rpm
df = df[(df['Revolution [rpm]'] >= n_min) & (df['Revolution [rpm]'] <= n_max)]

# Calculate torque
def calculate_torque_wrapper(row):
    working_pressure = row['Working pressure [bar]']
    current_speed = row['Revolution [rpm]']
    torque_constant = 0.14376997
    n1 = 25.7

    if current_speed < n1:
        torque = working_pressure * torque_constant
    else:
        torque = (n1 / current_speed) * torque_constant * working_pressure

    return round(torque, 2)

df['Calculated torque [kNm]'] = df.apply(calculate_torque_wrapper, axis=1)

def calculate_whisker_and_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    whisker_length = 1.5 * IQR
    lower_whisker = Q1 - whisker_length
    upper_whisker = Q3 + whisker_length
    outliers = data[(data < lower_whisker) | (data > upper_whisker)]
    return lower_whisker, upper_whisker, outliers

# Calculate whiskers and outliers for both torque and RPM
torque_lower_whisker, torque_upper_whisker, torque_outliers = calculate_whisker_and_outliers(df['Calculated torque [kNm]'])
rpm_lower_whisker, rpm_upper_whisker, rpm_outliers = calculate_whisker_and_outliers(df['Revolution [rpm]'])

# Anomaly detection
anomaly_threshold = 250  # bar
df['Is_Anomaly'] = df['Working pressure [bar]'] >= anomaly_threshold

# Create the plot
fig, ax = plt.subplots(figsize=(14, 10))

# Plot torque curves
ax.plot(rpm_curve[rpm_curve <= elbow_rpm_cont],
        np.full_like(rpm_curve[rpm_curve <= elbow_rpm_cont], M_cont_value),
        'g-', linewidth=2, label='M cont Max [kNm]')

ax.plot(rpm_curve[rpm_curve <= elbow_rpm_max],
        np.full_like(rpm_curve[rpm_curve <= elbow_rpm_max], M_max_Vg1),
        'r-', linewidth=2, label='M max Vg1 [kNm]')

ax.plot(rpm_curve[rpm_curve <= n_max], M_max_Vg2(rpm_curve[rpm_curve <= n_max]),
        'r--', linewidth=2, label='M max Vg2 [kNm]')

# Add vertical lines at the elbow points
ax.plot([elbow_rpm_max, elbow_rpm_max], [0, M_max_Vg1], color='purple', linestyle=':', linewidth=3)
ax.plot([elbow_rpm_cont, elbow_rpm_cont], [0, M_cont_value], color='orange', linestyle=':', linewidth=3)

# Add a truncated vertical line at n_max
ax.plot([n_max, n_max], [0, M_cont_value], color='black', linestyle='--', linewidth=2)

# Plot calculated torque vs RPM, differentiating between normal, anomaly, and outlier points
normal_data = df[(~df['Is_Anomaly']) & (~df['Calculated torque [kNm]'].isin(torque_outliers))]
anomaly_data = df[df['Is_Anomaly']]
torque_outlier_data = df[df['Calculated torque [kNm]'].isin(torque_outliers) & (~df['Is_Anomaly'])]
rpm_outlier_data = df[df['Revolution [rpm]'].isin(rpm_outliers) & (~df['Is_Anomaly'])]

scatter_normal = ax.scatter(normal_data['Revolution [rpm]'], normal_data['Calculated torque [kNm]'],
                            c=normal_data['Calculated torque [kNm]'], cmap='viridis',
                            s=50, alpha=0.6, label='Normal Data')
scatter_anomaly = ax.scatter(anomaly_data['Revolution [rpm]'], anomaly_data['Calculated torque [kNm]'],
                             color='red', s=100, alpha=0.8, marker='X', label='Anomaly (Pressure ≥ 250 bar)')
scatter_torque_outliers = ax.scatter(torque_outlier_data['Revolution [rpm]'], torque_outlier_data['Calculated torque [kNm]'],
                                     color='orange', s=100, alpha=0.8, marker='D', label='Torque Outliers')
scatter_rpm_outliers = ax.scatter(rpm_outlier_data['Revolution [rpm]'], rpm_outlier_data['Calculated torque [kNm]'],
                                  color='purple', s=100, alpha=0.8, marker='s', label='RPM Outliers')

# Add horizontal lines for the torque whiskers
ax.axhline(y=torque_upper_whisker, color='gray', linestyle='--', linewidth=1, label='Torque Upper Whisker')
ax.axhline(y=torque_lower_whisker, color='gray', linestyle=':', linewidth=1, label='Torque Lower Whisker')

# Set plot limits and labels
ax.set_xlim(0, x_axis_max)
ax.set_ylim(0, max(60, df['Calculated torque [kNm]'].max() * 1.1))
ax.set_xlabel('Drehzahl / speed / vitesse / revolutiones [1/min]')
ax.set_ylabel('Drehmoment / torque / couple / par de giro [kNm]')
plt.title('AVN800 DA975 - Hard Rock Test - 132kW (with Anomaly Detection)')

# Add grid
ax.grid(True, which='both', linestyle=':', color='gray', alpha=0.5)

# Add text annotations
ax.text(elbow_rpm_max * 0.5, M_max_Vg1 * 1.05, f'M max (max.): {M_max_Vg1} kNm',
        fontsize=10, ha='center', va='bottom', color='red')
ax.text(elbow_rpm_cont * 0.5, M_cont_value * 0.95, f'M cont (max.): {M_cont_value} kNm',
        fontsize=10, ha='center', va='top', color='green')

# Add text annotations for elbow points and n_max
ax.text(elbow_rpm_max, 0, f'{elbow_rpm_max:.2f}', ha='right', va='bottom', color='purple', fontsize=8)
ax.text(elbow_rpm_cont, 0, f'{elbow_rpm_cont:.2f}', ha='right', va='bottom', color='orange', fontsize=8)
ax.text(n_max, M_cont_value, f'Max RPM: {n_max}', ha='right', va='top', color='black', fontsize=8, rotation=90)

# Add colorbar for the scatter plot
cbar = plt.colorbar(scatter_normal)
cbar.set_label('Calculated Torque [kNm]')

# Add legend
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=10)

plt.tight_layout()
plt.show()

# Print statistics about the data
print("\nRPM Statistics:")
print(df['Revolution [rpm]'].describe())
print("\nCalculated Torque Statistics:")
print(df['Calculated torque [kNm]'].describe())
print("\nWorking Pressure Statistics:")
print(df['Working pressure [bar]'].describe())
print("\nAnomaly Detection Results:")
print(f"Total data points: {len(df)}")
print(f"Normal data points: {len(normal_data)}")
print(f"Anomaly data points: {len(anomaly_data)}")
print(f"Percentage of anomalies: {len(anomaly_data) / len(df) * 100:.2f}%")
print(f"\nElbow point Max (intersection of M max Vg1 and M max Vg2): {elbow_rpm_max:.2f} rpm")
print(f"Elbow point Cont (intersection of M cont Max and M max Vg2): {elbow_rpm_cont:.2f} rpm")
print("\nWhisker and Outlier Information:")
print(f"Torque Upper Whisker: {torque_upper_whisker:.2f} kNm")
print(f"Torque Lower Whisker: {torque_lower_whisker:.2f} kNm")
print(f"Number of torque outliers: {len(torque_outliers)}")
print(f"Percentage of torque outliers: {len(torque_outliers) / len(df) * 100:.2f}%")
print(f"\nRPM Upper Whisker: {rpm_upper_whisker:.2f} rpm")
print(f"RPM Lower Whisker: {rpm_lower_whisker:.2f} rpm")
print(f"Number of RPM outliers: {len(rpm_outliers)}")
print(f"Percentage of RPM outliers: {len(rpm_outliers) / len(df) * 100:.2f}%")
