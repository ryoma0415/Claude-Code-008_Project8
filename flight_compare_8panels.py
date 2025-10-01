#!/usr/bin/env python3
"""
Flight Data 8-Panel Comparison - 8つのグラフを2x4で比較表示
2D座標, 指令角度, PID合計, X軸PID, Y軸PID, マーカー数, 送信成功, 制御有効
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

class FlightCompare8Panels:
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
        """8パネルアニメーション作成"""
        # 大きめの図を作成（2行4列）
        fig = plt.figure(figsize=(28, 12))
        fig.suptitle('Flight Data 8-Panel Analysis', fontsize=16, y=0.98)

        # データサブセット作成
        data_subset = self.data[::self.skip_frames]
        n_frames = len(data_subset)
        time_data = data_subset['elapsed_time']

        # 1. 2D座標変動（1,1）
        ax1 = fig.add_subplot(2, 4, 1)
        ax1.set_title('2D Trajectory (X-Y)')
        ax1.set_xlabel('X [m]')
        ax1.set_ylabel('Y [m]')
        margin = 0.1
        ax1.set_xlim([self.data['pos_x'].min()-margin, self.data['pos_x'].max()+margin])
        ax1.set_ylim([self.data['pos_y'].min()-margin, self.data['pos_y'].max()+margin])
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.scatter(0, 0, c='r', s=150, marker='*', zorder=5)

        # 2. 指令角度[deg]（1,2）
        ax2 = fig.add_subplot(2, 4, 2)
        ax2.set_title('Command Angles [deg]')
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Angle [deg]')
        ax2.set_xlim([0, time_data.max()])
        y_min = min(self.data['roll_ref_deg'].min(), self.data['pitch_ref_deg'].min())
        y_max = max(self.data['roll_ref_deg'].max(), self.data['pitch_ref_deg'].max())
        ax2.set_ylim([y_min - 1, y_max + 1])
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        # 3. PID合計値（1,3）
        ax3 = fig.add_subplot(2, 4, 3)
        ax3.set_title('PID Total Output')
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('PID Output')
        ax3.set_xlim([0, time_data.max()])
        y_min = min(self.data['pid_x_total'].min(), self.data['pid_y_total'].min())
        y_max = max(self.data['pid_x_total'].max(), self.data['pid_y_total'].max())
        ax3.set_ylim([y_min - 0.1, y_max + 0.1])
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        # 4. X軸PID成分（1,4）
        ax4 = fig.add_subplot(2, 4, 4)
        ax4.set_title('X-axis PID Components')
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('PID Components')
        ax4.set_xlim([0, time_data.max()])
        y_min = self.data[['pid_x_p', 'pid_x_i', 'pid_x_d']].min().min()
        y_max = self.data[['pid_x_p', 'pid_x_i', 'pid_x_d']].max().max()
        ax4.set_ylim([y_min - 0.1, y_max + 0.1])
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        # 5. Y軸PID成分（2,1）
        ax5 = fig.add_subplot(2, 4, 5)
        ax5.set_title('Y-axis PID Components')
        ax5.set_xlabel('Time [s]')
        ax5.set_ylabel('PID Components')
        ax5.set_xlim([0, time_data.max()])
        y_min = self.data[['pid_y_p', 'pid_y_i', 'pid_y_d']].min().min()
        y_max = self.data[['pid_y_p', 'pid_y_i', 'pid_y_d']].max().max()
        ax5.set_ylim([y_min - 0.1, y_max + 0.1])
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        # 6. マーカー数（2,2）
        ax6 = fig.add_subplot(2, 4, 6)
        ax6.set_title('Detected Markers')
        ax6.set_xlabel('Time [s]')
        ax6.set_ylabel('Marker Count')
        ax6.set_xlim([0, time_data.max()])
        ax6.set_ylim([self.data['marker_count'].min()-0.5, self.data['marker_count'].max()+0.5])
        ax6.grid(True, alpha=0.3)

        # 7. 送信成功フラグ（2,3）
        ax7 = fig.add_subplot(2, 4, 7)
        ax7.set_title('Send Success Status')
        ax7.set_xlabel('Time [s]')
        ax7.set_ylabel('Status (0/1)')
        ax7.set_xlim([0, time_data.max()])
        ax7.set_ylim([-0.1, 1.1])
        ax7.grid(True, alpha=0.3)
        ax7.set_yticks([0, 1])
        ax7.set_yticklabels(['Fail', 'Success'])

        # 8. 制御有効フラグ（2,4）
        ax8 = fig.add_subplot(2, 4, 8)
        ax8.set_title('Control Active Status')
        ax8.set_xlabel('Time [s]')
        ax8.set_ylabel('Status (0/1)')
        ax8.set_xlim([0, time_data.max()])
        ax8.set_ylim([-0.1, 1.1])
        ax8.grid(True, alpha=0.3)
        ax8.set_yticks([0, 1])
        ax8.set_yticklabels(['Inactive', 'Active'])

        # アニメーション用オブジェクト作成
        # Panel 1 - 2D
        line2d, = ax1.plot([], [], 'b-', linewidth=0.8, alpha=0.5)
        point2d, = ax1.plot([], [], 'ro', markersize=8)
        trail2d = ax1.scatter([], [], c=[], s=10, cmap='viridis', vmin=0, vmax=1, alpha=0.6)

        # Panel 2 - Command Angles
        line_roll, = ax2.plot([], [], 'b-', label='Roll', linewidth=1)
        line_pitch, = ax2.plot([], [], 'r-', label='Pitch', linewidth=1)
        point_roll, = ax2.plot([], [], 'bo', markersize=5)
        point_pitch, = ax2.plot([], [], 'ro', markersize=5)
        ax2.legend(loc='upper right', fontsize=8)

        # Panel 3 - PID Total
        line_pid_x_total, = ax3.plot([], [], 'b-', label='X-axis', linewidth=1)
        line_pid_y_total, = ax3.plot([], [], 'r-', label='Y-axis', linewidth=1)
        point_pid_x_total, = ax3.plot([], [], 'bo', markersize=5)
        point_pid_y_total, = ax3.plot([], [], 'ro', markersize=5)
        ax3.legend(loc='upper right', fontsize=8)

        # Panel 4 - X PID Components
        line_x_p, = ax4.plot([], [], 'b-', label='P', linewidth=1)
        line_x_i, = ax4.plot([], [], 'r-', label='I', linewidth=1)
        line_x_d, = ax4.plot([], [], 'g-', label='D', linewidth=1)
        ax4.legend(loc='upper right', fontsize=8)

        # Panel 5 - Y PID Components
        line_y_p, = ax5.plot([], [], 'b-', label='P', linewidth=1)
        line_y_i, = ax5.plot([], [], 'r-', label='I', linewidth=1)
        line_y_d, = ax5.plot([], [], 'g-', label='D', linewidth=1)
        ax5.legend(loc='upper right', fontsize=8)

        # Panel 6 - Marker Count
        line_markers, = ax6.plot([], [], 'g-', linewidth=1.5)
        point_markers, = ax6.plot([], [], 'go', markersize=6)
        fill_markers = ax6.fill_between([], [], 0, alpha=0.3, color='g')

        # Panel 7 - Send Success
        line_send, = ax7.plot([], [], 'b-', linewidth=1.5, drawstyle='steps-post')
        fill_send = ax7.fill_between([], [], 0, alpha=0.3, color='b', step='post')

        # Panel 8 - Control Active
        line_control, = ax8.plot([], [], 'orange', linewidth=1.5, drawstyle='steps-post')
        fill_control = ax8.fill_between([], [], 0, alpha=0.3, color='orange', step='post')

        # 時間表示
        time_text = fig.text(0.02, 0.95, '', transform=fig.transFigure, fontsize=12)

        # fill_betweenオブジェクトを保持するための変数
        self.fill_objects = {'markers': None, 'send': None, 'control': None}

        def init():
            """初期化関数"""
            line2d.set_data([], [])
            point2d.set_data([], [])
            trail2d.set_offsets(np.empty((0, 2)))

            line_roll.set_data([], [])
            line_pitch.set_data([], [])
            point_roll.set_data([], [])
            point_pitch.set_data([], [])

            line_pid_x_total.set_data([], [])
            line_pid_y_total.set_data([], [])
            point_pid_x_total.set_data([], [])
            point_pid_y_total.set_data([], [])

            line_x_p.set_data([], [])
            line_x_i.set_data([], [])
            line_x_d.set_data([], [])

            line_y_p.set_data([], [])
            line_y_i.set_data([], [])
            line_y_d.set_data([], [])

            line_markers.set_data([], [])
            point_markers.set_data([], [])

            line_send.set_data([], [])
            line_control.set_data([], [])

            time_text.set_text('')

            return (line2d, point2d, trail2d, line_roll, line_pitch, point_roll, point_pitch,
                   line_pid_x_total, line_pid_y_total, point_pid_x_total, point_pid_y_total,
                   line_x_p, line_x_i, line_x_d, line_y_p, line_y_i, line_y_d,
                   line_markers, point_markers, line_send, line_control, time_text)

        def update(frame):
            """フレーム更新関数"""
            if frame < n_frames:
                # Panel 1 - 2D Trajectory
                line2d.set_data(data_subset['pos_x'][:frame+1],
                               data_subset['pos_y'][:frame+1])
                point2d.set_data([data_subset.iloc[frame]['pos_x']],
                                [data_subset.iloc[frame]['pos_y']])

                # Trail with color gradient
                if frame > 0:
                    positions = np.c_[data_subset['pos_x'][:frame+1],
                                     data_subset['pos_y'][:frame+1]]
                    colors = np.linspace(0, 1, frame+1)
                    trail2d.set_offsets(positions)
                    trail2d.set_array(colors)

                # Panel 2 - Command Angles
                line_roll.set_data(time_data[:frame+1],
                                  data_subset['roll_ref_deg'][:frame+1])
                line_pitch.set_data(time_data[:frame+1],
                                   data_subset['pitch_ref_deg'][:frame+1])
                if frame > 0:
                    point_roll.set_data([time_data.iloc[frame]],
                                       [data_subset.iloc[frame]['roll_ref_deg']])
                    point_pitch.set_data([time_data.iloc[frame]],
                                        [data_subset.iloc[frame]['pitch_ref_deg']])

                # Panel 3 - PID Total
                line_pid_x_total.set_data(time_data[:frame+1],
                                         data_subset['pid_x_total'][:frame+1])
                line_pid_y_total.set_data(time_data[:frame+1],
                                         data_subset['pid_y_total'][:frame+1])
                if frame > 0:
                    point_pid_x_total.set_data([time_data.iloc[frame]],
                                               [data_subset.iloc[frame]['pid_x_total']])
                    point_pid_y_total.set_data([time_data.iloc[frame]],
                                               [data_subset.iloc[frame]['pid_y_total']])

                # Panel 4 - X PID Components
                line_x_p.set_data(time_data[:frame+1],
                                 data_subset['pid_x_p'][:frame+1])
                line_x_i.set_data(time_data[:frame+1],
                                 data_subset['pid_x_i'][:frame+1])
                line_x_d.set_data(time_data[:frame+1],
                                 data_subset['pid_x_d'][:frame+1])

                # Panel 5 - Y PID Components
                line_y_p.set_data(time_data[:frame+1],
                                 data_subset['pid_y_p'][:frame+1])
                line_y_i.set_data(time_data[:frame+1],
                                 data_subset['pid_y_i'][:frame+1])
                line_y_d.set_data(time_data[:frame+1],
                                 data_subset['pid_y_d'][:frame+1])

                # Panel 6 - Marker Count
                line_markers.set_data(time_data[:frame+1],
                                     data_subset['marker_count'][:frame+1])
                if frame > 0:
                    point_markers.set_data([time_data.iloc[frame]],
                                          [data_subset.iloc[frame]['marker_count']])

                    # Fill area under curve
                    if self.fill_objects['markers']:
                        self.fill_objects['markers'].remove()
                    self.fill_objects['markers'] = ax6.fill_between(
                        time_data[:frame+1],
                        data_subset['marker_count'][:frame+1],
                        0, alpha=0.3, color='g'
                    )

                # Panel 7 - Send Success
                line_send.set_data(time_data[:frame+1],
                                  data_subset['send_success'][:frame+1])
                if self.fill_objects['send']:
                    self.fill_objects['send'].remove()
                self.fill_objects['send'] = ax7.fill_between(
                    time_data[:frame+1],
                    data_subset['send_success'][:frame+1],
                    0, alpha=0.3, color='b', step='post'
                )

                # Panel 8 - Control Active
                line_control.set_data(time_data[:frame+1],
                                     data_subset['control_active'][:frame+1])
                if self.fill_objects['control']:
                    self.fill_objects['control'].remove()
                self.fill_objects['control'] = ax8.fill_between(
                    time_data[:frame+1],
                    data_subset['control_active'][:frame+1],
                    0, alpha=0.3, color='orange', step='post'
                )

                # Time text
                current_time = data_subset.iloc[frame]['elapsed_time']
                time_text.set_text(f'Time: {current_time:.2f}s / Frame: {frame}/{n_frames}')

            return (line2d, point2d, trail2d, line_roll, line_pitch, point_roll, point_pitch,
                   line_pid_x_total, line_pid_y_total, point_pid_x_total, point_pid_y_total,
                   line_x_p, line_x_i, line_x_d, line_y_p, line_y_i, line_y_d,
                   line_markers, point_markers, line_send, line_control, time_text)

        # アニメーション作成
        interval = 1000 / (self.original_fps / self.skip_frames)
        anim = animation.FuncAnimation(
            fig, update, init_func=init,
            frames=n_frames, interval=interval, blit=False  # blitをFalseに設定（fill_betweenのため）
        )

        plt.tight_layout()
        return fig, anim, n_frames

    def save_animation(self, anim, fig, filename, total_frames):
        """アニメーションを保存"""
        print(f"\nアニメーション保存中... ({total_frames}フレーム)")

        # FFmpegライターの設定（ビットレートを高めに設定）
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='FlightCompare8Panels'), bitrate=4800)

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

        print("\n8パネル比較アニメーションを作成中...")
        fig, anim, n_frames = self.create_animation()

        # 保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"compare_8panels_{timestamp}.mp4"

        if not os.path.exists("animation_results"):
            os.makedirs("animation_results")
        filepath = os.path.join("animation_results", filename)

        self.save_animation(anim, fig, filepath, n_frames)
        plt.close(fig)

        print(f"\n動画ファイルが保存されました: {filepath}")

def main():
    """メイン関数"""
    print("\n飛行データ8パネル比較ツール")
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

    comparator = FlightCompare8Panels(csv_path)
    comparator.run()

if __name__ == "__main__":
    main()