#!/usr/bin/env python3
"""
DDoS Attack Detection using Regression Analysis
Web Server Log File Analysis - OPTIMIZED VERSION
Author: Naniko Meisrishvili
Date: February 12, 2026
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# Configure matplotlib visualizations - USE AGG BACKEND FOR FASTER PROCESSING
plt.switch_backend('Agg')
plt.rcParams['figure.figsize'] = [15, 8]
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


class DDoSAnalyzer:
    """DDoS attack detection using regression analysis"""

    def __init__(self, log_path):
        self.log_path = log_path
        self.df = None
        self.time_series = None
        self.baseline_stats = None
        self.attack_intervals = []

    def parse_log_file(self):
        """Parse web server log format with ISO timestamp - OPTIMIZED"""

        log_pattern = re.compile(
            r'(\S+) - - \[(.*?)\] "(.*?)" (\d+) (\d+)'
        )

        data = []
        print(f"📂 Reading log file: {self.log_path}")

        try:
            with open(self.log_path, 'r', encoding='utf-8', errors='ignore') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    if not line:
                        continue

                    match = log_pattern.match(line)
                    if match:
                        ip, timestamp_str, request, status, size = match.groups()

                        try:
                            # Fast timestamp parsing - take first 19 chars only
                            timestamp = datetime.strptime(timestamp_str[:19], '%Y-%m-%d %H:%M:%S')

                            # Parse request quickly
                            request_parts = request.split()
                            method = request_parts[0] if len(request_parts) > 0 else '-'
                            url = request_parts[1] if len(request_parts) > 1 else '-'

                            data.append({
                                'ip': ip,
                                'timestamp': timestamp,
                                'method': method,
                                'url': url,
                                'status': int(status),
                                'size': int(size)
                            })
                        except:
                            continue

                    # Progress indicator every 10,000 lines
                    if line_num % 10000 == 0:
                        print(f"   Processed {line_num:,} lines...")

        except FileNotFoundError:
            print(f"❌ Error: Log file '{self.log_path}' not found!")
            return None

        if not data:
            print("❌ No data was parsed from the log file!")
            return None

        self.df = pd.DataFrame(data)
        self.df = self.df.sort_values('timestamp')

        print(f"✅ Successfully parsed {len(self.df):,} log entries")
        print(f"📅 Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        print(f"🌐 Unique IPs: {self.df['ip'].nunique():,}")

        return self.df

    def aggregate_time_series(self, freq='1T'):
        """Aggregate requests by time interval - OPTIMIZED"""
        if self.df is None or len(self.df) == 0:
            print("❌ No data to aggregate!")
            return None

        try:
            # Fast aggregation using pandas
            self.df['time_bin'] = self.df['timestamp'].dt.floor(freq)
            self.time_series = self.df.groupby('time_bin', observed=True).size().reset_index()
            self.time_series.columns = ['timestamp', 'request_count']

            # Create complete time range
            full_range = pd.date_range(
                start=self.time_series['timestamp'].min(),
                end=self.time_series['timestamp'].max(),
                freq=freq
            )

            # Fast reindexing
            self.time_series = self.time_series.set_index('timestamp')
            self.time_series = self.time_series.reindex(full_range, fill_value=0)
            self.time_series = self.time_series.reset_index()
            self.time_series.columns = ['timestamp', 'request_count']

            total_duration = self.time_series['timestamp'].max() - self.time_series['timestamp'].min()
            total_hours = total_duration.total_seconds() / 3600

            print(f"📊 Time Series Statistics:")
            print(f"   • Total intervals: {len(self.time_series):,}")
            print(f"   • Time range: {total_hours:.1f} hours")
            print(f"   • Total requests: {self.time_series['request_count'].sum():,}")
            print(f"   • Average rate: {self.time_series['request_count'].mean():.2f} req/min")
            print(f"   • Max rate: {self.time_series['request_count'].max():.0f} req/min")

        except Exception as e:
            print(f"❌ Error aggregating time series: {e}")
            return None

        return self.time_series

    def calculate_baseline(self):
        """Calculate baseline statistics for normal traffic - OPTIMIZED"""
        if self.time_series is None:
            return None

        try:
            # Remove top 1% for baseline calculation (faster with quantile)
            threshold_99 = self.time_series['request_count'].quantile(0.99)
            normal_traffic = self.time_series[self.time_series['request_count'] <= threshold_99]['request_count']

            self.baseline_stats = {
                'mean': normal_traffic.mean(),
                'std': normal_traffic.std(),
                'median': self.time_series['request_count'].median(),
                'q95': self.time_series['request_count'].quantile(0.95),
                'q99': self.time_series['request_count'].quantile(0.99),
                'max': self.time_series['request_count'].max()
            }

            print("\n" + "=" * 50)
            print("📊 BASELINE TRAFFIC STATISTICS")
            print("=" * 50)
            print(f"   Normal Mean: {self.baseline_stats['mean']:.2f} req/min")
            print(f"   Normal Std:  {self.baseline_stats['std']:.2f}")
            print(f"   95th %ile:   {self.baseline_stats['q95']:.2f} req/min")
            print(f"   99th %ile:   {self.baseline_stats['q99']:.2f} req/min")
            print(f"   Max Rate:    {self.baseline_stats['max']:.0f} req/min")
            print("=" * 50)

        except Exception as e:
            print(f"❌ Error calculating baseline: {e}")
            return None

        return self.baseline_stats

    def polynomial_regression_analysis(self, degree=4):
        """Perform polynomial regression - OPTIMIZED for speed"""
        try:
            # Use every 10th point for large datasets to speed up regression
            if len(self.time_series) > 1000:
                sample_idx = np.arange(0, len(self.time_series), 10)
                X = sample_idx.reshape(-1, 1)
                y = self.time_series['request_count'].iloc[sample_idx].values
            else:
                X = np.arange(len(self.time_series)).reshape(-1, 1)
                y = self.time_series['request_count'].values

            # Create polynomial features
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_poly = poly.fit_transform(X)

            # Fit Ridge regression
            model = Ridge(alpha=1.0)
            model.fit(X_poly, y)

            # Predict for all points
            X_all = np.arange(len(self.time_series)).reshape(-1, 1)
            X_all_poly = poly.transform(X_all)
            y_pred = model.predict(X_all_poly)

            # Calculate residuals
            y_actual = self.time_series['request_count'].values
            residuals = y_actual - y_pred
            residual_std = np.std(residuals)

            # Detect anomalies
            anomalies = np.abs(residuals) > 3 * residual_std

            # Calculate metrics
            r2 = r2_score(y_actual, y_pred)
            rmse = np.sqrt(mean_squared_error(y_actual, y_pred))

            print("\n" + "=" * 50)
            print("📈 REGRESSION MODEL PERFORMANCE")
            print("=" * 50)
            print(f"   R² Score:           {r2:.4f}")
            print(f"   RMSE:               {rmse:.2f}")
            print(f"   Anomalies detected: {np.sum(anomalies)}")
            print("=" * 50)

            return {
                'predictions': y_pred,
                'residuals': residuals,
                'residual_std': residual_std,
                'anomalies': anomalies,
                'r2_score': r2,
                'rmse': rmse
            }

        except Exception as e:
            print(f"❌ Error in regression analysis: {e}")
            return None

    def detect_attack_intervals(self, threshold_multiplier=3):
        """Detect DDoS attack time intervals - OPTIMIZED"""
        if self.baseline_stats is None:
            self.calculate_baseline()
            if self.baseline_stats is None:
                return [], 0

        threshold = self.baseline_stats['mean'] + threshold_multiplier * self.baseline_stats['std']

        # Vectorized detection
        attack_mask = self.time_series['request_count'] > threshold

        # Find continuous attack periods
        attack_periods = []
        in_attack = False
        attack_start = None

        for i, (idx, row) in enumerate(self.time_series.iterrows()):
            if attack_mask.iloc[i]:
                if not in_attack:
                    attack_start = row['timestamp']
                    in_attack = True
            else:
                if in_attack:
                    attack_data = self.time_series[attack_mask &
                                                   (self.time_series['timestamp'] >= attack_start) &
                                                   (self.time_series['timestamp'] <= row['timestamp'])]

                    if len(attack_data) > 1:  # More than 1 minute
                        attack_periods.append({
                            'start': attack_start,
                            'end': row['timestamp'],
                            'duration': row['timestamp'] - attack_start,
                            'peak_rate': attack_data['request_count'].max(),
                            'avg_rate': attack_data['request_count'].mean(),
                            'total_requests': attack_data['request_count'].sum(),
                            'intensity': attack_data['request_count'].max() / self.baseline_stats['mean']
                        })
                    in_attack = False

        # Handle attack at the end
        if in_attack:
            attack_data = self.time_series[attack_mask &
                                           (self.time_series['timestamp'] >= attack_start)]
            if len(attack_data) > 1:
                attack_periods.append({
                    'start': attack_start,
                    'end': self.time_series['timestamp'].iloc[-1],
                    'duration': self.time_series['timestamp'].iloc[-1] - attack_start,
                    'peak_rate': attack_data['request_count'].max(),
                    'avg_rate': attack_data['request_count'].mean(),
                    'total_requests': attack_data['request_count'].sum(),
                    'intensity': attack_data['request_count'].max() / self.baseline_stats['mean']
                })

        self.attack_intervals = attack_periods

        print("\n" + "=" * 50)
        print(f"🚨 DDoS ATTACK DETECTION RESULTS")
        print("=" * 50)
        print(f"   Threshold: {threshold:.2f} req/min")
        print(f"   Attacks detected: {len(attack_periods)}")
        print("=" * 50)

        return attack_periods, threshold

    def create_visualizations(self, regression_results, threshold):
        """Create visualizations - OPTIMIZED with fewer points"""
        if self.time_series is None or regression_results is None:
            return None

        try:
            # Sample data for plotting (max 500 points for performance)
            if len(self.time_series) > 500:
                plot_idx = np.linspace(0, len(self.time_series) - 1, 500, dtype=int)
                plot_data = self.time_series.iloc[plot_idx]
                plot_predictions = regression_results['predictions'][plot_idx]
                plot_residuals = regression_results['residuals'][plot_idx]
                plot_anomalies = regression_results['anomalies'][plot_idx]
            else:
                plot_data = self.time_series
                plot_predictions = regression_results['predictions']
                plot_residuals = regression_results['residuals']
                plot_anomalies = regression_results['anomalies']

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # Reduced to 4 plots for speed
            fig.suptitle('DDoS Attack Detection Analysis', fontsize=14, fontweight='bold')

            # 1. Time series with threshold
            ax1 = axes[0, 0]
            ax1.plot(plot_data['timestamp'], plot_data['request_count'],
                     'b-', alpha=0.6, label='Traffic', linewidth=0.8)
            ax1.axhline(y=threshold, color='r', linestyle='--',
                        label=f'Threshold ({threshold:.0f})', linewidth=1.5)
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Requests/min')
            ax1.set_title('Traffic Pattern')
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.2)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, fontsize=8)

            # 2. Regression fit
            ax2 = axes[0, 1]
            ax2.plot(plot_data['timestamp'], plot_data['request_count'],
                     'b-', alpha=0.4, label='Actual', linewidth=0.8)
            ax2.plot(plot_data['timestamp'], plot_predictions,
                     'r-', label=f'Regression (R²={regression_results["r2_score"]:.3f})',
                     linewidth=1.5)
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Requests/min')
            ax2.set_title('Regression Fit')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.2)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, fontsize=8)

            # 3. Attack classification
            ax3 = axes[1, 0]
            normal_mask = plot_data['request_count'] <= threshold
            attack_mask = plot_data['request_count'] > threshold

            ax3.scatter(plot_data.loc[normal_mask, 'timestamp'],
                        plot_data.loc[normal_mask, 'request_count'],
                        c='blue', alpha=0.4, s=5, label='Normal')
            ax3.scatter(plot_data.loc[attack_mask, 'timestamp'],
                        plot_data.loc[attack_mask, 'request_count'],
                        c='red', alpha=0.6, s=10, label='Attack')
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Requests/min')
            ax3.set_title('Attack Classification')
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.2)
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, fontsize=8)

            # 4. Attack summary
            ax4 = axes[1, 1]
            ax4.axis('off')

            summary = f"""
            📊 SUMMARY
            {'─' * 30}

            📁 Log Entries: {len(self.df):,}
            📅 Date: {self.df['timestamp'].min().strftime('%Y-%m-%d')}

            📈 Baseline: {self.baseline_stats['mean']:.1f}±{self.baseline_stats['std']:.1f}/min
            🚨 Threshold: {threshold:.1f}/min

            🔴 Attacks: {len(self.attack_intervals)}
            """

            if self.attack_intervals:
                attack = self.attack_intervals[0]  # Show first attack
                summary += f"""

            ⚠️  Worst Attack:
               • {attack['start'].strftime('%H:%M')}-{attack['end'].strftime('%H:%M')}
               • Peak: {attack['peak_rate']:.0f}/min
               • {attack['intensity']:.1f}x normal
                """

            ax4.text(0.1, 0.95, summary, transform=ax4.transAxes,
                     fontsize=9, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

            plt.tight_layout()
            plt.savefig('ddos_analysis_dashboard.png', dpi=120, bbox_inches='tight')
            plt.close(fig)  # Don't display, just save
            print("✅ Dashboard saved as 'ddos_analysis_dashboard.png'")

        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")

        return None

    def exploratory_analysis(self):
        """Fast exploratory analysis - OPTIMIZED for 80k+ lines"""
        if self.df is None or len(self.df) == 0:
            return None

        try:
            print("   Creating exploratory visualizations...")

            # Sample data for faster plotting
            if len(self.df) > 5000:
                sample_df = self.df.sample(n=5000, random_state=42)
            else:
                sample_df = self.df

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # Reduced to 4 plots

            # 1. Top 10 IPs (fast)
            top_ips = self.df['ip'].value_counts().head(10)
            axes[0, 0].barh(range(len(top_ips)), top_ips.values, color='coral', alpha=0.7)
            axes[0, 0].set_yticks(range(len(top_ips)))
            axes[0, 0].set_yticklabels([ip[:15] + '...' if len(ip) > 15 else ip for ip in top_ips.index])
            axes[0, 0].set_title('Top 10 IP Addresses')
            axes[0, 0].set_xlabel('Requests')

            # 2. Status codes
            status_dist = self.df['status'].value_counts().sort_index()
            colors = ['green' if s < 400 else 'orange' if s < 500 else 'red' for s in status_dist.index]
            axes[0, 1].bar(status_dist.index.astype(str), status_dist.values, color=colors, alpha=0.7)
            axes[0, 1].set_title('HTTP Status Codes')
            axes[0, 1].set_xlabel('Status')
            axes[0, 1].set_ylabel('Count')

            # 3. HTTP methods
            method_dist = self.df['method'].value_counts()
            axes[1, 0].pie(method_dist.values[:4], labels=method_dist.index[:4], autopct='%1.1f%%')
            axes[1, 0].set_title('HTTP Methods')

            # 4. Hourly traffic (using all data but fast)
            hourly = self.df.groupby(self.df['timestamp'].dt.hour).size()
            axes[1, 1].plot(hourly.index, hourly.values, 'b-', marker='o', markersize=3)
            axes[1, 1].set_title('Hourly Traffic Pattern')
            axes[1, 1].set_xlabel('Hour')
            axes[1, 1].set_ylabel('Requests')
            axes[1, 1].grid(True, alpha=0.2)

            plt.tight_layout()
            plt.savefig('exploratory_analysis.png', dpi=120, bbox_inches='tight')
            plt.close(fig)
            print("✅ Exploratory analysis saved as 'exploratory_analysis.png'")

        except Exception as e:
            print(f"❌ Error in exploratory analysis: {e}")

        return None


def main():
    """Main execution function"""
    print("\n" + "=" * 60)
    print("   DDoS ATTACK DETECTION USING REGRESSION ANALYSIS")
    print("=" * 60 + "\n")

    # Log file path
    log_file = r'D:\proj\cda01\n_meisrishvili25_87421_server.log'

    # Initialize analyzer
    analyzer = DDoSAnalyzer(log_file)

    import time
    start_time = time.time()

    # Step 1: Parse log file
    print("\n📋 STEP 1: Parsing Log File")
    print("-" * 40)
    df = analyzer.parse_log_file()
    if df is None:
        return

    # Step 2: Aggregate by minute intervals
    print("\n📋 STEP 2: Time Series Aggregation")
    print("-" * 40)
    time_series = analyzer.aggregate_time_series('1T')
    if time_series is None:
        return

    # Step 3: Calculate baseline statistics
    print("\n📋 STEP 3: Baseline Traffic Analysis")
    print("-" * 40)
    baseline = analyzer.calculate_baseline()
    if baseline is None:
        return

    # Step 4: Perform regression analysis
    print("\n📋 STEP 4: Polynomial Regression Analysis")
    print("-" * 40)
    reg_results = analyzer.polynomial_regression_analysis(degree=4)
    if reg_results is None:
        return

    # Step 5: Detect DDoS attack intervals
    print("\n📋 STEP 5: DDoS Attack Detection")
    print("-" * 40)
    attack_periods, threshold = analyzer.detect_attack_intervals(threshold_multiplier=3)

    # Step 6: Exploratory Analysis (FAST)
    print("\n📋 STEP 6: Exploratory Data Analysis")
    print("-" * 40)
    analyzer.exploratory_analysis()

    # Step 7: Create visualizations (FAST)
    print("\n📋 STEP 7: Creating Analysis Dashboard")
    print("-" * 40)
    analyzer.create_visualizations(reg_results, threshold)

    # Final Results
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("   DDoS ATTACK DETECTION - FINAL RESULTS")
    print("=" * 60)

    if attack_periods:
        print(f"\n🚨 CRITICAL: {len(attack_periods)} DDoS Attack Period(s) Detected!")
        print("-" * 40)

        # Show top 5 most intense attacks
        attack_periods.sort(key=lambda x: x['intensity'], reverse=True)

        for i, attack in enumerate(attack_periods[:5], 1):
            print(f"\nAttack #{i}:")
            print(f"  📅 {attack['start'].strftime('%Y-%m-%d %H:%M:%S')} - {attack['end'].strftime('%H:%M:%S')}")
            print(f"  ⏱️  Duration: {attack['duration']}, Peak: {attack['peak_rate']:.0f}/min")
            print(f"  ⚡ {attack['intensity']:.1f}x normal traffic")

        if len(attack_periods) > 5:
            print(f"\n  ... and {len(attack_periods) - 5} more attacks")
    else:
        print("\n✅ No DDoS attacks detected")

    print(f"\n⏱️  Total execution time: {elapsed_time:.1f} seconds")
    print("=" * 60)
    print("✅ Analysis Complete!")
    print("📁 Output: exploratory_analysis.png, ddos_analysis_dashboard.png")
    print("=" * 60)


if __name__ == "__main__":
    main()