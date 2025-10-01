#!/usr/bin/env python3
"""
Flight Data 6-Panel Comparison - 6つのグラフを2x3で比較表示
3D座標変動, 2D座標変動, 指令角度[deg], PID合計値, X軸PID, Y軸PID
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import os
import glob
from tqdm import tqdm
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

class FlightCompare6Panels:
    def __init__(self, csv_path=None):
        self.csv_path = csv_path
        self.data = None
        self.original_fps = 100
        self.skip_frames = 2  # データ量削減のため2フレームごと

    def load_data(self):
        """CSVファイルからデータを読み込み"""
        if not self.csv_path:
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

            # PID合計値を計算
            self.data['pid_x_total'] = self.data['pid_x_p'] + self.data['pid_x_i'] + self.data['pid_x_d']
            self.data['pid_y_total'] = self.data['pid_y_p'] + self.data['pid_y_i'] + self.data['pid_y_d']

            # サンプリングレート計算
            time_diff = self.data['elapsed_time'].diff().median()
            self.original_fps = 1.0 / time_diff if time_diff > 0 else 100
            print(f"  サンプリングレート: {self.original_fps:.1f} Hz")

            return True
        except Exception as e:
            print(f"エラー: データ読み込み失敗 - {e}")
            return False

    def create_animation(self):
        """6パネルアニメーション作成"""
        # 大きめの図を作成（2行3列）
        fig = plt.figure(figsize=(24, 12))
        fig.suptitle('Flight Data 6-Panel Analysis', fontsize=16, y=0.98)

        # データサブセット作成
        data_subset = self.data[::self.skip_frames]
        n_frames = len(data_subset)
        time_data = data_subset['elapsed_time']

        # 1. 3D座標変動（左上）
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.set_title('3D Trajectory')
        ax1.set_xlabel('X [m]')
        ax1.set_ylabel('Y [m]')
        ax1.set_zlabel('Z [m]')
        ax1.set_xlim([self.data['pos_x'].min()-0.1, self.data['pos_x'].max()+0.1])
        ax1.set_ylim([self.data['pos_y'].min()-0.1, self.data['pos_y'].max()+0.1])
        ax1.set_zlim([self.data['pos_z'].min()-0.1, self.data['pos_z'].max()+0.1])
        ax1.scatter(0, 0, 0, c='r', s=100, marker='*', label='Target')

        # 2. 2D座標変動（中央上）
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.set_title('2D Trajectory (X-Y)')
        ax2.set_xlabel('X [m]')
        ax2.set_ylabel('Y [m]')
        margin = 0.1
        ax2.set_xlim([self.data['pos_x'].min()-margin, self.data['pos_x'].max()+margin])
        ax2.set_ylim([self.data['pos_y'].min()-margin, self.data['pos_y'].max()+margin])
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        ax2.scatter(0, 0, c='r', s=200, marker='*', zorder=5)

        # 3. 指令角度[deg]（右上）
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.set_title('Command Angles [deg]')
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Angle [deg]')
        ax3.set_xlim([0, time_data.max()])
        y_min = min(self.data['roll_ref_deg'].min(), self.data['pitch_ref_deg'].min())
        y_max = max(self.data['roll_ref_deg'].max(), self.data['pitch_ref_deg'].max())
        ax3.set_ylim([y_min - 1, y_max + 1])
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        # 4. PID合計値（左下）
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.set_title('PID Total Output')
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('PID Output')
        ax4.set_xlim([0, time_data.max()])
        y_min = min(self.data['pid_x_total'].min(), self.data['pid_y_total'].min())
        y_max = max(self.data['pid_x_total'].max(), self.data['pid_y_total'].max())
        ax4.set_ylim([y_min - 0.1, y_max + 0.1])
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        # 5. X軸PID成分（中央下）
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.set_title('X-axis PID Components')
        ax5.set_xlabel('Time [s]')
        ax5.set_ylabel('PID Components')
        ax5.set_xlim([0, time_data.max()])
        y_min = self.data[['pid_x_p', 'pid_x_i', 'pid_x_d']].min().min()
        y_max = self.data[['pid_x_p', 'pid_x_i', 'pid_x_d']].max().max()
        ax5.set_ylim([y_min - 0.1, y_max + 0.1])
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        # 6. Y軸PID成分（右下）
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.set_title('Y-axis PID Components')
        ax6.set_xlabel('Time [s]')
        ax6.set_ylabel('PID Components')
        ax6.set_xlim([0, time_data.max()])
        y_min = self.data[['pid_y_p', 'pid_y_i', 'pid_y_d']].min().min()
        y_max = self.data[['pid_y_p', 'pid_y_i', 'pid_y_d']].max().max()
        ax6.set_ylim([y_min - 0.1, y_max + 0.1])
        ax6.grid(True, alpha=0.3)
        ax6.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        # アニメーション用オブジェクト作成
        # Panel 1 - 3D
        line3d, = ax1.plot([], [], [], 'b-', linewidth=0.8)
        point3d, = ax1.plot([], [], [], 'ro', markersize=6)

        # Panel 2 - 2D
        line2d, = ax2.plot([], [], 'b-', linewidth=0.8, alpha=0.5)
        point2d, = ax2.plot([], [], 'ro', markersize=8)

        # Panel 3 - Command Angles
        line_roll, = ax3.plot([], [], 'b-', label='Roll', linewidth=1)
        line_pitch, = ax3.plot([], [], 'r-', label='Pitch', linewidth=1)
        ax3.legend(loc='upper right')

        # Panel 4 - PID Total
        line_pid_x_total, = ax4.plot([], [], 'b-', label='X-axis', linewidth=1)
        line_pid_y_total, = ax4.plot([], [], 'r-', label='Y-axis', linewidth=1)
        ax4.legend(loc='upper right')

        # Panel 5 - X PID Components
        line_x_p, = ax5.plot([], [], 'b-', label='P', linewidth=1)
        line_x_i, = ax5.plot([], [], 'r-', label='I', linewidth=1)
        line_x_d, = ax5.plot([], [], 'g-', label='D', linewidth=1)
        ax5.legend(loc='upper right')

        # Panel 6 - Y PID Components
        line_y_p, = ax6.plot([], [], 'b-', label='P', linewidth=1)
        line_y_i, = ax6.plot([], [], 'r-', label='I', linewidth=1)
        line_y_d, = ax6.plot([], [], 'g-', label='D', linewidth=1)
        ax6.legend(loc='upper right')

        # 時間表示
        time_text = fig.text(0.02, 0.95, '', transform=fig.transFigure, fontsize=12)

        def init():
            """初期化関数"""
            line3d.set_data([], [])
            line3d.set_3d_properties([])
            point3d.set_data([], [])
            point3d.set_3d_properties([])

            line2d.set_data([], [])
            point2d.set_data([], [])

            line_roll.set_data([], [])
            line_pitch.set_data([], [])

            line_pid_x_total.set_data([], [])
            line_pid_y_total.set_data([], [])

            line_x_p.set_data([], [])
            line_x_i.set_data([], [])
            line_x_d.set_data([], [])

            line_y_p.set_data([], [])
            line_y_i.set_data([], [])
            line_y_d.set_data([], [])

            time_text.set_text('')

            return (line3d, point3d, line2d, point2d, line_roll, line_pitch,
                   line_pid_x_total, line_pid_y_total, line_x_p, line_x_i, line_x_d,
                   line_y_p, line_y_i, line_y_d, time_text)

        def update(frame):
            """フレーム更新関数"""
            if frame < n_frames:
                # Panel 1 - 3D
                line3d.set_data(data_subset['pos_x'][:frame+1],
                               data_subset['pos_y'][:frame+1])
                line3d.set_3d_properties(data_subset['pos_z'][:frame+1])
                point3d.set_data([data_subset.iloc[frame]['pos_x']],
                                [data_subset.iloc[frame]['pos_y']])
                point3d.set_3d_properties([data_subset.iloc[frame]['pos_z']])

                # Panel 2 - 2D
                line2d.set_data(data_subset['pos_x'][:frame+1],
                               data_subset['pos_y'][:frame+1])
                point2d.set_data([data_subset.iloc[frame]['pos_x']],
                                [data_subset.iloc[frame]['pos_y']])

                # Panel 3 - Command Angles
                line_roll.set_data(time_data[:frame+1],
                                  data_subset['roll_ref_deg'][:frame+1])
                line_pitch.set_data(time_data[:frame+1],
                                   data_subset['pitch_ref_deg'][:frame+1])

                # Panel 4 - PID Total
                line_pid_x_total.set_data(time_data[:frame+1],
                                         data_subset['pid_x_total'][:frame+1])
                line_pid_y_total.set_data(time_data[:frame+1],
                                         data_subset['pid_y_total'][:frame+1])

                # Panel 5 - X PID Components
                line_x_p.set_data(time_data[:frame+1],
                                 data_subset['pid_x_p'][:frame+1])
                line_x_i.set_data(time_data[:frame+1],
                                 data_subset['pid_x_i'][:frame+1])
                line_x_d.set_data(time_data[:frame+1],
                                 data_subset['pid_x_d'][:frame+1])

                # Panel 6 - Y PID Components
                line_y_p.set_data(time_data[:frame+1],
                                 data_subset['pid_y_p'][:frame+1])
                line_y_i.set_data(time_data[:frame+1],
                                 data_subset['pid_y_i'][:frame+1])
                line_y_d.set_data(time_data[:frame+1],
                                 data_subset['pid_y_d'][:frame+1])

                # Time text
                current_time = data_subset.iloc[frame]['elapsed_time']
                time_text.set_text(f'Time: {current_time:.2f}s')

            return (line3d, point3d, line2d, point2d, line_roll, line_pitch,
                   line_pid_x_total, line_pid_y_total, line_x_p, line_x_i, line_x_d,
                   line_y_p, line_y_i, line_y_d, time_text)

        # アニメーション作成
        interval = 1000 / (self.original_fps / self.skip_frames)
        anim = animation.FuncAnimation(
            fig, update, init_func=init,
            frames=n_frames, interval=interval, blit=True
        )

        return fig, anim, n_frames

    def save_animation(self, anim, fig, filename, total_frames):
        """アニメーションを保存"""
        print(f"\nアニメーション保存中... ({total_frames}フレーム)")

        # FFmpegライターの設定
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='FlightCompare6Panels'), bitrate=3600)

        # プログレスバー付きで保存
        with tqdm(total=total_frames, desc="保存進捗", unit="frame") as pbar:
            def progress_callback(current_frame, total_frames):
                if current_frame > pbar.n:
                    pbar.update(current_frame - pbar.n)
                return True

            try:
                anim.save(filename, writer=writer, progress_callback=progress_callback)
            except TypeError:
                anim.save(filename, writer=writer)
                pbar.update(total_frames - pbar.n)

        print(f"✓ 保存完了: {filename}")

    def run(self):
        """メイン実行関数"""
        if not self.load_data():
            return

        print("\n6パネル比較アニメーションを作成中...")
        fig, anim, n_frames = self.create_animation()

        # 保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"compare_6panels_{timestamp}.mp4"

        if not os.path.exists("animation_results"):
            os.makedirs("animation_results")
        filepath = os.path.join("animation_results", filename)

        self.save_animation(anim, fig, filepath, n_frames)
        plt.close(fig)

        print(f"\n動画ファイルが保存されました: {filepath}")

def main():
    """メイン関数"""
    print("\n飛行データ6パネル比較ツール")
    print("-"*35)

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

    comparator = FlightCompare6Panels(csv_path)
    comparator.run()

if __name__ == "__main__":
    main()