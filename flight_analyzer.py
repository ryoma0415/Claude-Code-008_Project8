#!/usr/bin/env python3
"""
Flight Data Analyzer - 静的グラフ生成ツール
CSVファイルから飛行データを読み込み、各種グラフを生成・保存
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.dates as mdates
from datetime import datetime
import os
import glob
import platform

# OS別フォント設定
system = platform.system()
if system == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'Hiragino Sans'
elif system == 'Windows':
    plt.rcParams['font.family'] = 'MS Gothic'
else:  # Linux
    plt.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams['axes.unicode_minus'] = False

class FlightAnalyzer:
    def __init__(self, csv_path=None):
        """
        初期化
        Args:
            csv_path: CSVファイルのパス（Noneの場合は最新ファイルを自動選択）
        """
        self.csv_path = csv_path
        self.data = None
        self.plot_options = {
            1: ("3D座標変動", self.plot_3d_trajectory),
            2: ("2D座標変動（X, Y）", self.plot_2d_trajectory),
            3: ("位置誤差", self.plot_position_error),
            4: ("指令角度 [rad]", self.plot_command_angle_rad),
            5: ("指令角度 [deg]", self.plot_command_angle_deg),
            6: ("X軸Y軸のPID成分合計値", self.plot_pid_total),
            7: ("X軸のPID成分各項", self.plot_pid_x_components),
            8: ("Y軸のPID成分各項", self.plot_pid_y_components),
            9: ("検出されたマーカー数", self.plot_marker_count),
            10: ("送信成功フラグ", self.plot_send_success),
            11: ("制御有効フラグ", self.plot_control_active),
            12: ("ループ実行時間 [ms]", self.plot_loop_time),
            13: ("全データ総合表示", self.plot_all_data)
        }
        
    def load_data(self):
        """CSVファイルからデータを読み込み"""
        if not self.csv_path:
            # 最新のログファイルを自動選択
            log_files = glob.glob("flight_logs/log_*.csv")
            if not log_files:
                print("エラー: flight_logsフォルダ内にログファイルが見つかりません")
                return False
            self.csv_path = max(log_files, key=os.path.getctime)
            print(f"最新ファイルを選択: {self.csv_path}")
        
        try:
            self.data = pd.read_csv(self.csv_path)
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            print(f"✓ データ読み込み完了: {len(self.data)}行")
            return True
        except Exception as e:
            print(f"エラー: データ読み込み失敗 - {e}")
            return False
    
    def apply_smoothing(self, data, window=5):
        """移動平均によるスムージング"""
        return data.rolling(window=window, center=True).mean()
    
    def plot_3d_trajectory(self, ax):
        """3D座標変動のプロット"""
        ax = plt.subplot(1, 1, 1, projection='3d')
        ax.plot(self.data['pos_x'], self.data['pos_y'], self.data['pos_z'], 
                'b-', linewidth=0.8, alpha=0.8)
        ax.scatter(0, 0, 0, c='r', s=100, marker='*', label='Target')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title('3D Trajectory')
        ax.legend()
        ax.grid(True)
        
    def plot_2d_trajectory(self, ax):
        """2D座標変動（X, Y）のプロット"""
        # 時間で色を変化させる
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.data)))
        ax.scatter(self.data['pos_x'], self.data['pos_y'], 
                   c=colors, s=1, alpha=0.5)
        ax.plot(self.data['pos_x'], self.data['pos_y'], 
                'gray', linewidth=0.3, alpha=0.3)
        ax.scatter(0, 0, c='r', s=200, marker='*', label='Target', zorder=5)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('2D Trajectory (XY Plane)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
    def plot_position_error(self, ax):
        """位置誤差のプロット"""
        time = self.data['elapsed_time']
        ax.plot(time, self.data['error_x'], 'b-', label='Error X', linewidth=0.8)
        ax.plot(time, self.data['error_y'], 'r-', label='Error Y', linewidth=0.8)
        error_magnitude = np.sqrt(self.data['error_x']**2 + self.data['error_y']**2)
        ax.plot(time, error_magnitude, 'g--', label='Magnitude', linewidth=0.8)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Position Error [m]')
        ax.set_title('Position Error over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def plot_command_angle_rad(self, ax):
        """指令角度[rad]のプロット"""
        time = self.data['elapsed_time']
        ax.plot(time, self.data['roll_ref_rad'], 'b-', label='Roll', linewidth=0.8)
        ax.plot(time, self.data['pitch_ref_rad'], 'r-', label='Pitch', linewidth=0.8)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Command Angle [rad]')
        ax.set_title('Command Angles (Radians)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def plot_command_angle_deg(self, ax):
        """指令角度[deg]のプロット"""
        time = self.data['elapsed_time']
        ax.plot(time, self.data['roll_ref_deg'], 'b-', label='Roll', linewidth=0.8)
        ax.plot(time, self.data['pitch_ref_deg'], 'r-', label='Pitch', linewidth=0.8)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Command Angle [deg]')
        ax.set_title('Command Angles (Degrees)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def plot_pid_total(self, ax):
        """X軸Y軸のPID成分合計値のプロット"""
        time = self.data['elapsed_time']
        x_total = self.data['pid_x_p'] + self.data['pid_x_i'] + self.data['pid_x_d']
        y_total = self.data['pid_y_p'] + self.data['pid_y_i'] + self.data['pid_y_d']
        ax.plot(time, x_total, 'b-', label='X-axis Total', linewidth=0.8)
        ax.plot(time, y_total, 'r-', label='Y-axis Total', linewidth=0.8)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('PID Output')
        ax.set_title('PID Total Output')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def plot_pid_x_components(self, ax):
        """X軸のPID成分各項のプロット"""
        time = self.data['elapsed_time']
        ax.plot(time, self.data['pid_x_p'], 'b-', label='P', linewidth=0.8)
        ax.plot(time, self.data['pid_x_i'], 'r-', label='I', linewidth=0.8)
        ax.plot(time, self.data['pid_x_d'], 'g-', label='D', linewidth=0.8)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('PID Components')
        ax.set_title('X-axis PID Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def plot_pid_y_components(self, ax):
        """Y軸のPID成分各項のプロット"""
        time = self.data['elapsed_time']
        ax.plot(time, self.data['pid_y_p'], 'b-', label='P', linewidth=0.8)
        ax.plot(time, self.data['pid_y_i'], 'r-', label='I', linewidth=0.8)
        ax.plot(time, self.data['pid_y_d'], 'g-', label='D', linewidth=0.8)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('PID Components')
        ax.set_title('Y-axis PID Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def plot_marker_count(self, ax):
        """検出されたマーカー数のプロット"""
        time = self.data['elapsed_time']
        ax.plot(time, self.data['marker_count'], 'b-', linewidth=0.8)
        ax.fill_between(time, 0, self.data['marker_count'], alpha=0.3)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Marker Count')
        ax.set_title('Detected Markers')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.5, max(4, self.data['marker_count'].max() + 0.5)])
        
    def plot_send_success(self, ax):
        """送信成功フラグのプロット"""
        time = self.data['elapsed_time']
        ax.plot(time, self.data['send_success'], 'g-', linewidth=0.8)
        ax.fill_between(time, 0, self.data['send_success'], alpha=0.3, color='green')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Send Success')
        ax.set_title('Command Send Status')
        ax.set_ylim([-0.1, 1.1])
        ax.grid(True, alpha=0.3)
        
    def plot_control_active(self, ax):
        """制御有効フラグのプロット"""
        time = self.data['elapsed_time']
        ax.plot(time, self.data['control_active'], 'b-', linewidth=0.8)
        ax.fill_between(time, 0, self.data['control_active'], alpha=0.3, color='blue')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Control Active')
        ax.set_title('Control Active Status')
        ax.set_ylim([-0.1, 1.1])
        ax.grid(True, alpha=0.3)
        
    def plot_loop_time(self, ax):
        """ループ実行時間[ms]のプロット"""
        time = self.data['elapsed_time']
        ax.plot(time, self.data['loop_time_ms'], 'b-', linewidth=0.8)
        ax.axhline(y=10, color='r', linestyle='--', linewidth=0.8, label='Target (10ms)')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Loop Time [ms]')
        ax.set_title('Control Loop Execution Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def plot_all_data(self, ax):
        """全データを1つの図に表示（サブプロット）"""
        fig = plt.figure(figsize=(16, 12))
        
        # 3D trajectory
        ax1 = fig.add_subplot(3, 4, 1, projection='3d')
        ax1.plot(self.data['pos_x'], self.data['pos_y'], self.data['pos_z'], 
                'b-', linewidth=0.5, alpha=0.8)
        ax1.set_xlabel('X[m]', fontsize=8)
        ax1.set_ylabel('Y[m]', fontsize=8)
        ax1.set_zlabel('Z[m]', fontsize=8)
        ax1.set_title('3D Trajectory', fontsize=10)
        
        # 2D trajectory
        ax2 = fig.add_subplot(3, 4, 2)
        self.plot_2d_trajectory(ax2)
        ax2.set_title('2D Trajectory', fontsize=10)
        
        # Position error
        ax3 = fig.add_subplot(3, 4, 3)
        self.plot_position_error(ax3)
        ax3.set_title('Position Error', fontsize=10)
        
        # Command angles
        ax4 = fig.add_subplot(3, 4, 4)
        self.plot_command_angle_deg(ax4)
        ax4.set_title('Command Angles', fontsize=10)
        
        # PID components X
        ax5 = fig.add_subplot(3, 4, 5)
        self.plot_pid_x_components(ax5)
        ax5.set_title('X-axis PID', fontsize=10)
        
        # PID components Y
        ax6 = fig.add_subplot(3, 4, 6)
        self.plot_pid_y_components(ax6)
        ax6.set_title('Y-axis PID', fontsize=10)
        
        # Marker count
        ax7 = fig.add_subplot(3, 4, 7)
        self.plot_marker_count(ax7)
        ax7.set_title('Markers', fontsize=10)
        
        # Loop time
        ax8 = fig.add_subplot(3, 4, 8)
        self.plot_loop_time(ax8)
        ax8.set_title('Loop Time', fontsize=10)
        
        # Control status
        ax9 = fig.add_subplot(3, 4, 9)
        time = self.data['elapsed_time']
        ax9.plot(time, self.data['control_active'], 'b-', label='Control')
        ax9.plot(time, self.data['send_success'], 'g-', label='Send', alpha=0.5)
        ax9.set_xlabel('Time [s]', fontsize=8)
        ax9.set_ylabel('Status', fontsize=8)
        ax9.set_title('System Status', fontsize=10)
        ax9.legend(fontsize=8)
        ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
        
    def show_menu(self):
        """メニューを表示して選択を取得"""
        print("\n" + "="*50)
        print("飛行データ解析 - グラフ選択メニュー")
        print("="*50)
        
        for key, (name, _) in self.plot_options.items():
            print(f"{key:2d}: {name}")
        
        print("\n複数選択する場合はカンマ区切りで入力（例: 1,3,5）")
        print("スムージングを適用する場合は最後に 's' を追加（例: 1,3,5s）")
        print("0: 終了")
        
    def run(self):
        """メインループ"""
        if not self.load_data():
            return
            
        while True:
            self.show_menu()
            choice = input("\n選択 > ").strip()
            
            if choice == '0':
                print("終了します")
                break
            
            # スムージング判定
            apply_smoothing = False
            if choice.endswith('s'):
                apply_smoothing = True
                choice = choice[:-1]
                print("スムージングを適用します")
            
            # 複数選択の処理
            try:
                selections = [int(x) for x in choice.split(',')]
            except:
                print("無効な入力です")
                continue
            
            # 無効な選択をフィルタ
            valid_selections = [s for s in selections if s in self.plot_options]
            
            if not valid_selections:
                print("有効な選択がありません")
                continue
            
            # スムージング適用
            if apply_smoothing:
                original_data = self.data.copy()
                numeric_columns = self.data.select_dtypes(include=[np.number]).columns
                for col in numeric_columns:
                    if col != 'elapsed_time':
                        self.data[col] = self.apply_smoothing(self.data[col])
            
            # グラフ作成
            if len(valid_selections) == 1:
                # 単一グラフ
                fig = plt.figure(figsize=(12, 8))
                
                if valid_selections[0] == 1:  # 3D plot特別処理
                    ax = fig.add_subplot(111, projection='3d')
                else:
                    ax = fig.add_subplot(111)
                
                name, plot_func = self.plot_options[valid_selections[0]]
                
                if valid_selections[0] == 13:  # 全データ表示
                    fig = plot_func(None)
                else:
                    plot_func(ax)
                    fig.suptitle(name)
            else:
                # 複数グラフ
                n_plots = len(valid_selections)
                cols = int(np.ceil(np.sqrt(n_plots)))
                rows = int(np.ceil(n_plots / cols))
                
                fig = plt.figure(figsize=(cols*5, rows*4))
                
                for idx, selection in enumerate(valid_selections):
                    if selection == 1:  # 3D plot
                        ax = fig.add_subplot(rows, cols, idx+1, projection='3d')
                    else:
                        ax = fig.add_subplot(rows, cols, idx+1)
                    
                    name, plot_func = self.plot_options[selection]
                    
                    if selection != 13:  # 全データ表示は除外
                        plot_func(ax)
                        ax.set_title(name, fontsize=10)
                
                plt.tight_layout()
            
            # スムージング復元
            if apply_smoothing:
                self.data = original_data
            
            # 保存と表示
            plt.show()
            
            save = input("グラフを保存しますか？ (y/n) > ").strip().lower()
            if save == 'y':
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"analysis_{timestamp}.png"
                if not os.path.exists("analysis_results"):
                    os.makedirs("analysis_results")
                filepath = os.path.join("analysis_results", filename)
                fig.savefig(filepath, dpi=150, bbox_inches='tight')
                print(f"✓ 保存完了: {filepath}")

def main():
    """メイン関数"""
    print("\n飛行データ解析ツール")
    print("-"*30)
    
    # CSVファイル選択
    print("\n1: 最新ファイルを自動選択")
    print("2: ファイルパスを指定")
    choice = input("選択 > ").strip()
    
    csv_path = None
    if choice == '2':
        csv_path = input("CSVファイルパス > ").strip()
        if not os.path.exists(csv_path):
            print(f"エラー: ファイルが見つかりません - {csv_path}")
            return
    
    analyzer = FlightAnalyzer(csv_path)
    analyzer.run()

if __name__ == "__main__":
    main()