import numpy as np
import plotly.graph_objects as go
from ipywidgets import interact, widgets, Layout
from IPython.display import display, HTML

def calculate_and_plot_gimbal(
    Mass_of_Nose_Cone, Length_of_Nose_Cone, Outer_Radius_of_Rocket,
    Mass_of_Rocket_Body_Tube, Length_of_Rocket_Body_Tube, Inner_Radius_of_Rocket_Body_Tube,
    Mass_of_F_Class_Motor, Length_of_F_Class_Motor, Outer_Radius_of_Motor, Inner_Radius_of_Motor_Core,
    Mass_of_Inner_Blue_Ring, Height_of_Inner_Blue_Ring, Inner_Radius_of_Inner_Blue_Ring, Outer_Radius_of_Inner_Blue_Ring,
    Mass_of_Single_Guide_Wheel, Outer_Radius_of_Guide_Wheel, Radial_Distance_of_Guide_Wheels,
    Mass_of_Outer_Gold_Ring, Height_of_Outer_Gold_Ring, Inner_Radius_of_Outer_Gold_Ring, Outer_Radius_of_Outer_Gold_Ring,
    Mass_of_Blue_U_Bracket, Center_of_Mass_Inertia_of_U_Bracket, Clearance_Distance_to_U_Bracket_Base,
    Offset_Distance_of_Grey_Rod, Rotation_Angle_of_Main_Grey_Rod, Tilt_Angle_of_Outer_Gold_Ring
):
    # ==========================================
    # 1. STANDALONE MULTI-BODY ROCKET BASELINE
    # ==========================================
    Center_of_Mass_Nose_Cone = 0.75 * Length_of_Nose_Cone
    Center_of_Mass_Body_Tube = Length_of_Nose_Cone + 0.5 * Length_of_Rocket_Body_Tube
    Center_of_Mass_Motor = Length_of_Nose_Cone + Length_of_Rocket_Body_Tube - 0.5 * Length_of_F_Class_Motor
    
    Total_Mass_of_Rocket = Mass_of_Nose_Cone + Mass_of_Rocket_Body_Tube + Mass_of_F_Class_Motor
    
    Combined_Center_of_Gravity_of_Rocket = (
        (Mass_of_Nose_Cone * Center_of_Mass_Nose_Cone) +
        (Mass_of_Rocket_Body_Tube * Center_of_Mass_Body_Tube) +
        (Mass_of_F_Class_Motor * Center_of_Mass_Motor)
    ) / Total_Mass_of_Rocket
    
    Inertia_Nose_Cone = Mass_of_Nose_Cone * ((3/20)*Outer_Radius_of_Rocket**2 + (3/80)*Length_of_Nose_Cone**2)
    Inertia_Body_Tube = (1/12) * Mass_of_Rocket_Body_Tube * (3*(Outer_Radius_of_Rocket**2 + Inner_Radius_of_Rocket_Body_Tube**2) + Length_of_Rocket_Body_Tube**2)
    Inertia_Motor = (1/12) * Mass_of_F_Class_Motor * (3*(Outer_Radius_of_Motor**2 + Inner_Radius_of_Motor_Core**2) + Length_of_F_Class_Motor**2)
    
    Baseline_Rocket_Lateral_Inertia = (
        (Inertia_Nose_Cone + Mass_of_Nose_Cone * (Center_of_Mass_Nose_Cone - Combined_Center_of_Gravity_of_Rocket)**2) +
        (Inertia_Body_Tube + Mass_of_Rocket_Body_Tube * (Center_of_Mass_Body_Tube - Combined_Center_of_Gravity_of_Rocket)**2) +
        (Inertia_Motor + Mass_of_F_Class_Motor * (Center_of_Mass_Motor - Combined_Center_of_Gravity_of_Rocket)**2)
    )
    
    # ==========================================
    # 2. GIMBAL OPERATIONAL MODES (AXIAL LOADS)
    # ==========================================
    Inertia_Rocket_Roll = 0.5 * Total_Mass_of_Rocket * Outer_Radius_of_Rocket**2
    Inertia_Inner_Blue_Ring_Roll = 0.5 * Mass_of_Inner_Blue_Ring * (Inner_Radius_of_Inner_Blue_Ring**2 + Outer_Radius_of_Inner_Blue_Ring**2)
    Effective_Inertia_of_Guide_Wheels = 2 * Mass_of_Single_Guide_Wheel * Outer_Radius_of_Inner_Blue_Ring**2
    Total_Mode_1_Roll_Inertia = Inertia_Rocket_Roll + Inertia_Inner_Blue_Ring_Roll + Effective_Inertia_of_Guide_Wheels
    
    Inertia_Inner_Blue_Ring_Lateral = Mass_of_Inner_Blue_Ring * ((1/12)*Height_of_Inner_Blue_Ring**2 + 0.25*(Inner_Radius_of_Inner_Blue_Ring**2 + Outer_Radius_of_Inner_Blue_Ring**2))
    Inertia_Outer_Gold_Ring_Lateral = Mass_of_Outer_Gold_Ring * ((1/12)*Height_of_Outer_Gold_Ring**2 + 0.25*(Inner_Radius_of_Outer_Gold_Ring**2 + Outer_Radius_of_Outer_Gold_Ring**2))
    Inertia_Guide_Wheels_Lateral = 4 * ((0.25 * Mass_of_Single_Guide_Wheel * Outer_Radius_of_Guide_Wheel**2) + Mass_of_Single_Guide_Wheel * Radial_Distance_of_Guide_Wheels**2)
    Total_Mode_2_Tilt_Inertia = Baseline_Rocket_Lateral_Inertia + Inertia_Inner_Blue_Ring_Lateral + Inertia_Outer_Gold_Ring_Lateral + Inertia_Guide_Wheels_Lateral
    
    # ==========================================
    # 3. DYNAMIC TENSOR STATE TRANSFORMATION
    # ==========================================
    Tilt_Angle_Radians = np.radians(Tilt_Angle_of_Outer_Gold_Ring)
    Main_Rod_Angle_Radians = np.radians(Rotation_Angle_of_Main_Grey_Rod)
    
    Shifted_Inertia_of_U_Bracket = Center_of_Mass_Inertia_of_U_Bracket + Mass_of_Blue_U_Bracket * Clearance_Distance_to_U_Bracket_Base**2
    Local_Inertia_of_Internal_Gimbal = Total_Mode_2_Tilt_Inertia * (np.cos(Tilt_Angle_Radians)**2) + Total_Mode_1_Roll_Inertia * (np.sin(Tilt_Angle_Radians)**2)
    
    Total_Internal_Moving_Mass = Total_Mass_of_Rocket + Mass_of_Inner_Blue_Ring + Mass_of_Outer_Gold_Ring + (4 * Mass_of_Single_Guide_Wheel)
    Shifted_Inertia_of_Internal_Gimbal = Local_Inertia_of_Internal_Gimbal + Total_Internal_Moving_Mass * Offset_Distance_of_Grey_Rod**2
    Total_Mode_3_Main_Drive_Inertia = Shifted_Inertia_of_U_Bracket + Shifted_Inertia_of_Internal_Gimbal
    
    # ==========================================
    # 4. COMPLIANCE ASSESSMENT ENGINE
    # ==========================================
    Percentage_Increase_Y = ((Total_Mode_2_Tilt_Inertia - Baseline_Rocket_Lateral_Inertia) / Baseline_Rocket_Lateral_Inertia) * 100\n",
    Percentage_Increase_X = ((Total_Mode_3_Main_Drive_Inertia - Baseline_Rocket_Lateral_Inertia) / Baseline_Rocket_Lateral_Inertia) * 100\n",
    
    Status_Y = "PASS" if Percentage_Increase_Y <= 5.0 else "FAIL"
    Status_X = "PASS" if Percentage_Increase_X <= 5.0 else "FAIL"
    Color_Y = "#2ecc71" if Status_Y == "PASS" else "#e74c3c"
    Color_X = "#2ecc71" if Status_X == "PASS" else "#e74c3c"
    
    Html_Panel = f"""
    <div style='border: 3px solid #34495e; padding: 20px; border-radius: 12px; background-color: #ffffff; font-family: Segoe UI, Arial, sans-serif; max-width: 600px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
        <h2 style='margin-top:0; color:#2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px;'>Gimbal Testbench Compliance Report</h2>
        <p style='font-size:14px; color:#7f8c8d;'><b>Baseline Rocket Lateral Inertia:</b> {Baseline_Rocket_Lateral_Inertia:.6f} kg·m²</p>
        <div style='background-color: #f8f9fa; padding: 12px; border-radius: 6px; margin-bottom: 10px; border-left: 5px solid {Color_Y};'>
            <h4 style='margin:0; color:#34495e;'>Horizontal Axis 2 (Y-Axis Pins / Tilt)</h4>\n",
            <p style='margin: 5px 0 0 0; font-size:14px;'>Total Inertia: <b>{Total_Mode_2_Tilt_Inertia:.6f} kg·m²</b></p>\n",
            <p style='margin: 3px 0 0 0; font-size:14px;'>Added Inertia: <span style='color:{Color_Y}; font-weight:bold;'>{Percentage_Increase_Y:.2f}% [{Status_Y}]</span> (Spec Limit: ≤ 5.0%)</p>\n",
        </div>
        <div style='background-color: #f8f9fa; padding: 12px; border-radius: 6px; border-left: 5px solid {Color_X};'>
            <h4 style='margin:0; color:#34495e;'>Horizontal Axis 3 (X-Axis Grey Rod / Main Drive)</h4>\n",
            <p style='margin: 5px 0 0 0; font-size:14px;'>Total Inertia at Selected Angle: <b>{Total_Mode_3_Main_Drive_Inertia:.6f} kg·m²</b></p>\n",
            <p style='margin: 3px 0 0 0; font-size:14px;'>Added Inertia: <span style='color:{Color_X}; font-weight:bold;'>{Percentage_Increase_X:.2f}% [{Status_X}]</span> (Spec Limit: ≤ 5.0%)</p>\n",
        </div>
    </div>
    """
    display(HTML(Html_Panel))
    
    # ==========================================
    # 5. KINEMATIC 3D RENDERING SYSTEM
    # ==========================================
    fig = go.Figure()
    Resolution = 25
    
    def transform_geometry(x, y, z):
        x_1 = x
        y_1 = y * np.cos(Tilt_Angle_Radians) - z * np.sin(Tilt_Angle_Radians)
        z_1 = y * np.sin(Tilt_Angle_Radians) + z * np.cos(Tilt_Angle_Radians)
        z_2 = z_1 + Offset_Distance_of_Grey_Rod
        x_final = x_1
        y_final = y_1 * np.cos(Main_Rod_Angle_Radians) - z_2 * np.sin(Main_Rod_Angle_Radians)
        z_final = y_1 * np.sin(Main_Rod_Angle_Radians) + z_2 * np.cos(Main_Rod_Angle_Radians)
        return x_final, y_final, z_final
    
    # Rocket Body Surface Mesh
    z_steps_body = np.linspace(-Length_of_Rocket_Body_Tube/2, Length_of_Rocket_Body_Tube/2, Resolution)
    radial_steps = np.linspace(0, 2*np.pi, Resolution)
    Mesh_Theta, Mesh_Z_Body = np.meshgrid(radial_steps, z_steps_body)
    Mesh_X_Body = Outer_Radius_of_Rocket * np.cos(Mesh_Theta)
    Mesh_Y_Body = Outer_Radius_of_Rocket * np.sin(Mesh_Theta)
    X_b, Y_b, Z_b = transform_geometry(Mesh_X_Body, Mesh_Y_Body, Mesh_Z_Body)
    fig.add_trace(go.Surface(x=X_b, y=Y_b, z=Z_b, colorscale=[[0, '#3498db'], [1, '#3498db']], showscale=False, opacity=0.85))
    
    # Inner Tracking Blue Ring Mesh
    z_steps_ring = np.linspace(-Height_of_Inner_Blue_Ring/2, Height_of_Inner_Blue_Ring/2, 10)
    Mesh_Theta_Ring, Mesh_Z_Ring = np.meshgrid(radial_steps, z_steps_ring)
    X_Blue_Ring = Outer_Radius_of_Inner_Blue_Ring * np.cos(Mesh_Theta_Ring)
    Y_Blue_Ring = Outer_Radius_of_Inner_Blue_Ring * np.sin(Mesh_Theta_Ring)
    X_br, Y_br, Z_br = transform_geometry(X_Blue_Ring, Y_Blue_Ring, Mesh_Z_Ring)
    fig.add_trace(go.Surface(x=X_br, y=Y_br, z=Z_br, colorscale=[[0, '#2980b9'], [1, '#2980b9']], showscale=False, opacity=0.4))
    
    # Outer Structure Gold Ring Mesh
    X_Gold_Ring = Outer_Radius_of_Outer_Gold_Ring * np.cos(Mesh_Theta_Ring)
    Y_Gold_Ring = Outer_Radius_of_Outer_Gold_Ring * np.sin(Mesh_Theta_Ring)
    X_gr, Y_gr, Z_gr = transform_geometry(X_Gold_Ring, Y_Gold_Ring, Mesh_Z_Ring * 1.05)
    fig.add_trace(go.Surface(x=X_gr, y=Y_gr, z=Z_gr, colorscale=[[0, '#f1c40f'], [1, '#f1c40f']], showscale=False, opacity=0.45))
    
    # Structural U-Bracket Polyline Line Setup
    Bracket_Radius = Outer_Radius_of_Outer_Gold_Ring + 0.05
    Bracket_Theta = np.linspace(-np.pi/2, np.pi/2, 20)
    Bracket_X = np.zeros_like(Bracket_Theta)
    Bracket_Y = Bracket_Radius * np.cos(Bracket_Theta)
    Bracket_Z = Bracket_Radius * np.sin(Bracket_Theta) + Offset_Distance_of_Grey_Rod
    U_X = Bracket_X
