import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    st.title("Torque Analysis App")
    st.sidebar.markdown("Created by Kursat Kilic - Geotechnical Digitalization")

    # Sidebar for user inputs
    st.sidebar.header("Parameter Settings")
    P_max = st.sidebar.number_input("Maximum power (kW)", value=132.0, min_value=1.0, max_value=500.0)
    nu = st.sidebar.number_input("Efficiency coefficient", value=0.7, min_value=0.1, max_value=1.0)
    n_max = st.sidebar.number_input("Maximum rpm", value=25.3, min_value=1.0, max_value=100.0)
    n_min = st.sidebar.number_input("Minimum rpm", value=21.0, min_value=1.0, max_value=n_max)
    x_axis_max = st.sidebar.number_input("X-axis maximum", value=26.0, min_value=n_max, max_value=100.0)
    M_cont_value = st.sidebar.number_input("Continuous torque (kNm)", value=44.0, min_value=1.0, max_value=100.0)
    M_max_Vg1 = st.sidebar.number_input("Maximum torque for Vg1 (kNm)", value=54.0, min_value=1.0, max_value=200.0)
    anomaly_threshold = st.sidebar.number_input("Anomaly threshold (bar)", value=250, min_value=100, max_value=500)

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file, sep=';', decimal=',')

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
        df['Is_Anomaly'] = df['Working pressure [bar]'] >= anomaly_threshold

        # Function to calculate M max Vg2
        def M_max_Vg2(rpm):
            return np.minimum(M_max_Vg1, (P_max * 60 * nu) / (2 * np.pi * rpm))

        # Calculate the intersection points
        elbow_rpm_max = (P_max * 60 * nu) / (2 * np.pi * M_max_Vg1)
        elbow_rpm_cont = (P_max * 60 * nu) / (2 * np.pi * M_cont_value)

        # Generate rpm values for the continuous curves
        rpm_curve = np.linspace(0.1, n_max, 1000)  # Avoid division by zero

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
                                     color='red', s=100, alpha=0.8, marker='X', label=f'Anomaly (Pressure ≥ {anomaly_threshold} bar)')
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
        st.pyplot(fig)

        # Display statistics
        st.subheader("Data Statistics")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("RPM Statistics:")
            st.write(df['Revolution [rpm]'].describe())

        with col2:
            st.write("Calculated Torque Statistics:")
            st.write(df['Calculated torque [kNm]'].describe())

        with col3:
            st.write("Working Pressure Statistics:")
            st.write(df['Working pressure [bar]'].describe())

        # Anomaly Detection Results
        st.subheader("Anomaly Detection Results")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write(f"Total data points: {len(df)}")
            st.write(f"Normal data points: {len(normal_data)}")
            st.write(f"Anomaly data points: {len(anomaly_data)}")
            st.write(f"Percentage of anomalies: {len(anomaly_data) / len(df) * 100:.2f}%")

        with col2:
            st.write(f"Elbow point Max: {elbow_rpm_max:.2f} rpm")
            st.write(f"Elbow point Cont: {elbow_rpm_cont:.2f} rpm")

        with col3:
            st.write("Whisker and Outlier Information:")
            st.write(f"Torque Upper Whisker: {torque_upper_whisker:.2f} kNm")
            st.write(f"Torque Lower Whisker: {torque_lower_whisker:.2f} kNm")
            st.write(f"Number of torque outliers: {len(torque_outliers)}")
            st.write(f"Percentage of torque outliers: {len(torque_outliers) / len(df) * 100:.2f}%")
            st.write(f"RPM Upper Whisker: {rpm_upper_whisker:.2f} rpm")
            st.write(f"RPM Lower Whisker: {rpm_lower_whisker:.2f} rpm")
            st.write(f"Number of RPM outliers: {len(rpm_outliers)}")
            st.write(f"Percentage of RPM outliers: {len(rpm_outliers) / len(df) * 100:.2f}%")

    else:
        st.info("Please upload a CSV file to begin the analysis.")

    # Add footer with creator information
    st.markdown("---")
    st.markdown("Created by Kursat Kilic - Geotechnical Digitalization")

if __name__ == "__main__":
    main()
