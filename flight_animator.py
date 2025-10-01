#!/usr/bin/env python3
"""
Flight Data Animator - アニメーション生成ツール
CSVファイルから飛行データを読み込み、各種アニメーションを生成・保存
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

class FlightAnimator:
    def __init__(self, csv_path=None):
        """
        初期化
        Args:
            csv_path: CSVファイルのパス（Noneの場合は最新ファイルを自動選択）
        """
        self.csv_path = csv_path
        self.data = None
        self.animation_options = {
            1: ("3D座標変動", self.animate_3d_trajectory),
            2: ("2D座標変動（X, Y）", self.animate_2d_trajectory),
            3: ("位置誤差", self.animate_position_error),
            4: ("指令角度 [rad]", self.animate_command_angle_rad),
            5: ("指令角度 [deg]", self.animate_command_angle_deg),
            6: ("X軸Y軸のPID成分合計値", self.animate_pid_total),
            7: ("X軸のPID成分各項", self.animate_pid_x_components),
            8: ("Y軸のPID成分各項", self.animate_pid_y_components),
            9: ("検出されたマーカー数", self.animate_marker_count),
            10: ("送信成功フラグ", self.animate_send_success),
            11: ("制御有効フラグ", self.animate_control_active),
            12: ("ループ実行時間 [ms]", self.animate_loop_time),
            13: ("全データ総合表示", self.animate_all_data)
        }
        
        # アニメーション設定
        self.speed_options = {
            '1': (1.0, "実時間再生"),
            '2': (2.0, "2倍速再生"),
            '3': (5.0, "5倍速再生")
        }
        self.speed_multiplier = 1.0
        self.skip_frames = 1
        
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
            
            # データのサンプリングレート計算
            time_diff = self.data['elapsed_time'].diff().median()
            self.original_fps = 1.0 / time_diff if time_diff > 0 else 100
            print(f"  サンプリングレート: {self.original_fps:.1f} Hz")
            
            return True
        except Exception as e:
            print(f"エラー: データ読み込み失敗 - {e}")
            return False
    
    def apply_smoothing(self, data, window=5):
        """移動平均によるスムージング"""
        return data.rolling(window=window, center=True).mean()
    
    def create_animation_base(self, fig, init_func, update_func, n_frames, interval):
        """アニメーションの基本作成"""
        anim = animation.FuncAnimation(
            fig, update_func, init_func=init_func,
            frames=n_frames, interval=interval, blit=False
        )
        return anim
    
    def animate_3d_trajectory(self):
        """3D座標変動のアニメーション"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # データ準備
        data_subset = self.data[::self.skip_frames]
        n_frames = len(data_subset)
        
        # 軸の範囲設定
        ax.set_xlim([self.data['pos_x'].min()-0.1, self.data['pos_x'].max()+0.1])
        ax.set_ylim([self.data['pos_y'].min()-0.1, self.data['pos_y'].max()+0.1])
        ax.set_zlim([self.data['pos_z'].min()-0.1, self.data['pos_z'].max()+0.1])
        
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title('3D Trajectory Animation')
        
        # ターゲット表示
        ax.scatter(0, 0, 0, c='r', s=100, marker='*', label='Target')
        
        # アニメーション用オブジェクト
        line, = ax.plot([], [], [], 'b-', linewidth=0.8)
        point, = ax.plot([], [], [], 'ro', markersize=8)
        time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes)
        
        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
            time_text.set_text('')
            return line, point, time_text
        
        def update(frame):
            # 軌跡
            line.set_data(data_subset['pos_x'][:frame+1], 
                         data_subset['pos_y'][:frame+1])
            line.set_3d_properties(data_subset['pos_z'][:frame+1])
            
            # 現在位置
            if frame < n_frames:
                point.set_data([data_subset.iloc[frame]['pos_x']], 
                              [data_subset.iloc[frame]['pos_y']])
                point.set_3d_properties([data_subset.iloc[frame]['pos_z']])
                time_text.set_text(f'Time: {data_subset.iloc[frame]["elapsed_time"]:.2f}s')
            
            return line, point, time_text
        
        interval = 1000 / (self.original_fps * self.speed_multiplier / self.skip_frames)
        return fig, init, update, n_frames, interval
    
    def animate_2d_trajectory(self):
        """2D座標変動のアニメーション"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # データ準備
        data_subset = self.data[::self.skip_frames]
        n_frames = len(data_subset)
        
        # 軸の設定
        margin = 0.1
        ax.set_xlim([self.data['pos_x'].min()-margin, self.data['pos_x'].max()+margin])
        ax.set_ylim([self.data['pos_y'].min()-margin, self.data['pos_y'].max()+margin])
        ax.set_aspect('equal')
        
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('2D Trajectory Animation')
        ax.grid(True, alpha=0.3)
        
        # ターゲット表示
        ax.scatter(0, 0, c='r', s=200, marker='*', label='Target', zorder=5)
        
        # アニメーション用オブジェクト
        line, = ax.plot([], [], 'b-', linewidth=0.8, alpha=0.5)
        points = ax.scatter([], [], c=[], s=20, cmap='viridis', vmin=0, vmax=1)
        current_point, = ax.plot([], [], 'ro', markersize=10)
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        
        def init():
            line.set_data([], [])
            points.set_offsets(np.empty((0, 2)))
            current_point.set_data([], [])
            time_text.set_text('')
            return line, points, current_point, time_text
        
        def update(frame):
            # 軌跡
            line.set_data(data_subset['pos_x'][:frame+1], 
                         data_subset['pos_y'][:frame+1])
            
            # 色付き点群（時間経過を色で表現）
            if frame > 0:
                positions = np.c_[data_subset['pos_x'][:frame+1], 
                                 data_subset['pos_y'][:frame+1]]
                colors = np.linspace(0, 1, frame+1)
                points.set_offsets(positions)
                points.set_array(colors)
            
            # 現在位置
            if frame < n_frames:
                current_point.set_data([data_subset.iloc[frame]['pos_x']], 
                                      [data_subset.iloc[frame]['pos_y']])
                time_text.set_text(f'Time: {data_subset.iloc[frame]["elapsed_time"]:.2f}s')
            
            return line, points, current_point, time_text
        
        interval = 1000 / (self.original_fps * self.speed_multiplier / self.skip_frames)
        return fig, init, update, n_frames, interval
    
    def animate_position_error(self):
        """位置誤差のアニメーション"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # データ準備
        data_subset = self.data[::self.skip_frames]
        n_frames = len(data_subset)
        time_data = data_subset['elapsed_time']
        
        # 誤差の大きさを計算
        data_subset['error_magnitude'] = np.sqrt(
            data_subset['error_x']**2 + data_subset['error_y']**2
        )
        
        # 軸の設定
        ax1.set_xlim([0, time_data.max()])
        ax1.set_ylim([min(self.data['error_x'].min(), self.data['error_y'].min())-0.05,
                      max(self.data['error_x'].max(), self.data['error_y'].max())+0.05])
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Position Error [m]')
        ax1.set_title('Position Error Animation')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        ax2.set_xlim([0, time_data.max()])
        ax2.set_ylim([0, data_subset['error_magnitude'].max()+0.05])
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Error Magnitude [m]')
        ax2.set_title('Error Magnitude')
        ax2.grid(True, alpha=0.3)
        
        # アニメーション用オブジェクト
        line_x, = ax1.plot([], [], 'b-', label='Error X', linewidth=1)
        line_y, = ax1.plot([], [], 'r-', label='Error Y', linewidth=1)
        line_mag, = ax2.plot([], [], 'g-', label='Magnitude', linewidth=1)
        
        point_x, = ax1.plot([], [], 'bo', markersize=8)
        point_y, = ax1.plot([], [], 'ro', markersize=8)
        point_mag, = ax2.plot([], [], 'go', markersize=8)
        
        ax1.legend(loc='upper right')
        ax2.legend(loc='upper right')
        
        time_text = fig.text(0.02, 0.96, '', transform=fig.transFigure)
        
        def init():
            line_x.set_data([], [])
            line_y.set_data([], [])
            line_mag.set_data([], [])
            point_x.set_data([], [])
            point_y.set_data([], [])
            point_mag.set_data([], [])
            time_text.set_text('')
            return line_x, line_y, line_mag, point_x, point_y, point_mag, time_text
        
        def update(frame):
            # ラインデータ
            line_x.set_data(time_data[:frame+1], data_subset['error_x'][:frame+1])
            line_y.set_data(time_data[:frame+1], data_subset['error_y'][:frame+1])
            line_mag.set_data(time_data[:frame+1], data_subset['error_magnitude'][:frame+1])
            
            # 現在位置マーカー
            if frame < n_frames:
                current_time = time_data.iloc[frame]
                point_x.set_data([current_time], [data_subset.iloc[frame]['error_x']])
                point_y.set_data([current_time], [data_subset.iloc[frame]['error_y']])
                point_mag.set_data([current_time], [data_subset.iloc[frame]['error_magnitude']])
                time_text.set_text(f'Time: {current_time:.2f}s')
            
            return line_x, line_y, line_mag, point_x, point_y, point_mag, time_text
        
        interval = 1000 / (self.original_fps * self.speed_multiplier / self.skip_frames)
        return fig, init, update, n_frames, interval
    
    def animate_simple_timeseries(self, columns, labels, title, ylabel):
        """汎用的な時系列アニメーション"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # データ準備
        data_subset = self.data[::self.skip_frames]
        n_frames = len(data_subset)
        time_data = data_subset['elapsed_time']
        
        # 軸の設定
        ax.set_xlim([0, time_data.max()])
        
        # Y軸の範囲を計算
        y_min = min([self.data[col].min() for col in columns])
        y_max = max([self.data[col].max() for col in columns])
        margin = (y_max - y_min) * 0.1
        ax.set_ylim([y_min - margin, y_max + margin])
        
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        # カラーマップ
        colors = plt.cm.tab10(np.linspace(0, 1, len(columns)))
        
        # アニメーション用オブジェクト
        lines = []
        points = []
        for i, (col, label, color) in enumerate(zip(columns, labels, colors)):
            line, = ax.plot([], [], color=color, label=label, linewidth=1)
            point, = ax.plot([], [], 'o', color=color, markersize=8)
            lines.append(line)
            points.append(point)
        
        ax.legend(loc='upper right')
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        
        def init():
            for line, point in zip(lines, points):
                line.set_data([], [])
                point.set_data([], [])
            time_text.set_text('')
            return lines + points + [time_text]
        
        def update(frame):
            for line, point, col in zip(lines, points, columns):
                line.set_data(time_data[:frame+1], data_subset[col][:frame+1])
                if frame < n_frames:
                    point.set_data([time_data.iloc[frame]], [data_subset.iloc[frame][col]])
            
            if frame < n_frames:
                time_text.set_text(f'Time: {time_data.iloc[frame]:.2f}s')
            
            return lines + points + [time_text]
        
        interval = 1000 / (self.original_fps * self.speed_multiplier / self.skip_frames)
        return fig, init, update, n_frames, interval
    
    def animate_command_angle_rad(self):
        """指令角度[rad]のアニメーション"""
        return self.animate_simple_timeseries(
            ['roll_ref_rad', 'pitch_ref_rad'],
            ['Roll', 'Pitch'],
            'Command Angles Animation [rad]',
            'Command Angle [rad]'
        )
    
    def animate_command_angle_deg(self):
        """指令角度[deg]のアニメーション"""
        return self.animate_simple_timeseries(
            ['roll_ref_deg', 'pitch_ref_deg'],
            ['Roll', 'Pitch'],
            'Command Angles Animation [deg]',
            'Command Angle [deg]'
        )
    
    def animate_pid_total(self):
        """PID成分合計値のアニメーション"""
        # PID合計を計算
        self.data['pid_x_total'] = self.data['pid_x_p'] + self.data['pid_x_i'] + self.data['pid_x_d']
        self.data['pid_y_total'] = self.data['pid_y_p'] + self.data['pid_y_i'] + self.data['pid_y_d']
        
        return self.animate_simple_timeseries(
            ['pid_x_total', 'pid_y_total'],
            ['X-axis Total', 'Y-axis Total'],
            'PID Total Output Animation',
            'PID Output'
        )
    
    def animate_pid_x_components(self):
        """X軸PID成分のアニメーション"""
        return self.animate_simple_timeseries(
            ['pid_x_p', 'pid_x_i', 'pid_x_d'],
            ['P', 'I', 'D'],
            'X-axis PID Components Animation',
            'PID Components'
        )
    
    def animate_pid_y_components(self):
        """Y軸PID成分のアニメーション"""
        return self.animate_simple_timeseries(
            ['pid_y_p', 'pid_y_i', 'pid_y_d'],
            ['P', 'I', 'D'],
            'Y-axis PID Components Animation',
            'PID Components'
        )
    
    def animate_marker_count(self):
        """マーカー数のアニメーション"""
        return self.animate_simple_timeseries(
            ['marker_count'],
            ['Marker Count'],
            'Detected Markers Animation',
            'Marker Count'
        )
    
    def animate_send_success(self):
        """送信成功フラグのアニメーション"""
        return self.animate_simple_timeseries(
            ['send_success'],
            ['Send Success'],
            'Command Send Status Animation',
            'Status (0/1)'
        )
    
    def animate_control_active(self):
        """制御有効フラグのアニメーション"""
        return self.animate_simple_timeseries(
            ['control_active'],
            ['Control Active'],
            'Control Active Status Animation',
            'Status (0/1)'
        )
    
    def animate_loop_time(self):
        """ループ実行時間のアニメーション"""
        return self.animate_simple_timeseries(
            ['loop_time_ms'],
            ['Loop Time'],
            'Control Loop Execution Time Animation',
            'Time [ms]'
        )
    
    def animate_all_data(self):
        """全データの総合アニメーション（実装省略）"""
        print("全データ総合アニメーションは処理が重いため、個別アニメーションをご利用ください")
        return None, None, None, 0, 0
    
    def save_animation(self, anim, fig, filename, total_frames):
        """アニメーションをMP4として保存（プログレスバー付き）"""
        print(f"\nアニメーション保存中... ({total_frames}フレーム)")

        # FFmpegライターの設定
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='FlightAnimator'), bitrate=1800)

        # プログレスバー付きで保存
        with tqdm(total=total_frames, desc="保存進捗", unit="frame") as pbar:
            # プログレスコールバック付きで保存
            def progress_callback(current_frame, total_frames):
                if current_frame > pbar.n:
                    pbar.update(current_frame - pbar.n)
                return True

            # 保存実行（progress_callbackをサポートする場合に備えて渡す）
            try:
                anim.save(filename, writer=writer, progress_callback=progress_callback)
            except TypeError:
                # progress_callbackがサポートされていない場合
                # フレームごとの書き込みで進捗を表示
                anim.save(filename, writer=writer)
                # 全フレーム完了として表示
                pbar.update(total_frames - pbar.n)

        print(f"✓ 保存完了: {filename}")
    
    def show_menu(self):
        """メニューを表示"""
        print("\n" + "="*50)
        print("飛行データアニメーション - メニュー")
        print("="*50)
        
        for key, (name, _) in self.animation_options.items():
            print(f"{key:2d}: {name}")
        
        print("\n0: 終了")
        
    def show_speed_menu(self):
        """速度設定メニュー"""
        print("\n再生速度を選択:")
        for key, (mult, name) in self.speed_options.items():
            print(f"{key}: {name}")
        
    def show_skip_menu(self):
        """フレームスキップ設定"""
        print("\nフレームスキップ設定:")
        print("1: スキップなし（全フレーム）")
        print("2: 2フレームごと（データ量1/2）")
        print("5: 5フレームごと（データ量1/5）")
        print("10: 10フレームごと（データ量1/10）")
        
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
            
            try:
                selection = int(choice)
                if selection not in self.animation_options:
                    print("無効な選択です")
                    continue
            except:
                print("無効な入力です")
                continue
            
            # 速度設定
            self.show_speed_menu()
            speed_choice = input("選択 (デフォルト: 1) > ").strip() or '1'
            if speed_choice in self.speed_options:
                self.speed_multiplier = self.speed_options[speed_choice][0]
                print(f"✓ {self.speed_options[speed_choice][1]}を設定")
            
            # スキップ設定
            self.show_skip_menu()
            skip_choice = input("選択 (デフォルト: 1) > ").strip() or '1'
            try:
                self.skip_frames = int(skip_choice)
                print(f"✓ {self.skip_frames}フレームごとに表示")
            except:
                self.skip_frames = 1
            
            # スムージング
            apply_smoothing = input("スムージングを適用しますか？ (y/n, デフォルト: n) > ").strip().lower()
            
            if apply_smoothing == 'y':
                original_data = self.data.copy()
                numeric_columns = self.data.select_dtypes(include=[np.number]).columns
                for col in numeric_columns:
                    if col != 'elapsed_time':
                        self.data[col] = self.apply_smoothing(self.data[col])
                print("✓ スムージングを適用")
            
            # アニメーション作成
            name, animate_func = self.animation_options[selection]
            print(f"\n「{name}」のアニメーションを作成中...")
            
            result = animate_func()
            if result[0] is None:
                continue
            
            fig, init, update, n_frames, interval = result
            
            # アニメーション生成
            print(f"フレーム数: {n_frames}, インターバル: {interval:.1f}ms")
            anim = self.create_animation_base(fig, init, update, n_frames, interval)

            # 自動保存（プレビュー表示なし）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            animation_name = name.replace(' ', '_').replace('/', '_')
            filename = f"animation_{animation_name}_{timestamp}.mp4"

            if not os.path.exists("animation_results"):
                os.makedirs("animation_results")
            filepath = os.path.join("animation_results", filename)

            self.save_animation(anim, fig, filepath, n_frames)
            print(f"\n動画ファイルが保存されました: {filepath}")
            
            # スムージング復元
            if apply_smoothing == 'y':
                self.data = original_data
            
            plt.close(fig)

def main():
    """メイン関数"""
    print("\n飛行データアニメーションツール")
    print("-"*35)
    
    # 依存関係チェック
    try:
        import matplotlib
        matplotlib.use('TkAgg')  # インタラクティブバックエンド
    except:
        print("警告: matplotlibの設定に問題があります")
    
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
    
    animator = FlightAnimator(csv_path)
    animator.run()

if __name__ == "__main__":
    main()