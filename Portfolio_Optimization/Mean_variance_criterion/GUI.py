import customtkinter as ctk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import minimize, Bounds, LinearConstraint
from Mean_variance_criterion import MV_criterion
import warnings
warnings.filterwarnings('ignore')

# --- Risk profile presets ---
RISK_PROFILES = {
    "Conservative": {"min_w": 0.0, "max_w": 0.2},
    "Balanced":      {"min_w": 0.01, "max_w": 0.3},
    "Aggressive":    {"min_w": 0.0, "max_w": 0.5},
}

class PortfolioOptimizerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("💼 Portfolio Optimizer")
        self.geometry("1500x900")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.X_MV = None
        self.test_set = None
        self.ticker_list = []

        self.period_options = ["1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"]
        self.interval_options = ["1d","1wk","1mo"]
        self.benchmark_options = {"NIFTY 50":"^NSEI","S&P 500":"^GSPC","NASDAQ":"^IXIC"}

        # --- Sidebar ---
        self.sidebar = ctk.CTkScrollableFrame(self, width=300, label_text="Controls")
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)

        # Ticker input
        self.ticker_frame = ctk.CTkFrame(self.sidebar)
        self.ticker_frame.pack(fill="x", pady=(10,5), padx=5)
        self.ticker_entries = []
        for _ in range(3):
            self.add_ticker_field()
        ctk.CTkButton(self.sidebar, text="+ Add Stock", command=self.add_ticker_field)\
            .pack(pady=5, padx=5, fill="x")

        # Period/interval/benchmark
        self.period_combo    = self.create_dropdown("Period",    self.period_options,  "5y")
        self.interval_combo  = self.create_dropdown("Interval",  self.interval_options,"1mo")
        self.benchmark_combo = self.create_dropdown("Benchmark", list(self.benchmark_options.keys()), "NIFTY 50")

        # Risk profile
        self.profile_combo = self.create_dropdown("Risk Profile", list(RISK_PROFILES.keys()), "Balanced")

        # Capital
        self.capital_entry = self.create_entry(self.sidebar, "Capital (₹)", "100000")

        # Action buttons
        ctk.CTkLabel(self.sidebar, text="Actions").pack(pady=(10,0))
        ctk.CTkButton(self.sidebar, text="Optimize Portfolio",     command=self.optimize_portfolio)   .pack(pady=5, fill="x", padx=5)
        ctk.CTkButton(self.sidebar, text="Show Cumulative Return",  command=self.plot_cumulative_return).pack(pady=5, fill="x", padx=5)
        ctk.CTkButton(self.sidebar, text="Export Weights to CSV",   command=self.export_to_csv)       .pack(pady=5, fill="x", padx=5)
        ctk.CTkButton(self.sidebar, text="Exit", fg_color="red", command=self.destroy)             .pack(pady=10, fill="x", padx=5)

        # --- Main area ---
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(expand=True, fill="both", padx=10, pady=10)

        cols = ("Ticker","Price","Weight","Allocation")
        self.tree = ttk.Treeview(self.main_frame, columns=cols, show="headings", height=6)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, anchor="center")
        self.tree.pack(fill="x", padx=10, pady=(10,5))

        self.metrics_label = ctk.CTkLabel(self.main_frame, text="", font=("Arial",14))
        self.metrics_label.pack(pady=5)

        # Plot area
        self.plot_frame = ctk.CTkFrame(self.main_frame)
        self.plot_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # News feed
        self.news_frame = ctk.CTkScrollableFrame(self.main_frame, label_text="News (top 3 per ticker)")
        self.news_frame.pack(fill="both", padx=10, pady=5, ipady=10)

        # Log console
        self.log_textbox = ctk.CTkTextbox(self.main_frame, height=120)
        self.log_textbox.pack(fill="x", padx=10, pady=10)
        self.log("Log initialized...")

    def add_ticker_field(self):
        entry = ctk.CTkEntry(self.ticker_frame, placeholder_text="Ticker")
        entry.pack(side="top", pady=2, padx=5, fill="x")
        self.ticker_entries.append(entry)

    def create_dropdown(self, label, options, default):
        ctk.CTkLabel(self.sidebar, text=label).pack(pady=(10,0))
        combo = ctk.CTkComboBox(self.sidebar, values=options)
        combo.set(default)
        combo.pack(pady=5, fill="x", padx=5)
        return combo

    def create_entry(self, parent, label, placeholder):
        ctk.CTkLabel(parent, text=label).pack(pady=(10,0))
        e = ctk.CTkEntry(parent, placeholder_text=placeholder)
        e.pack(pady=5, padx=5, fill="x")
        return e

    def optimize_portfolio(self):
        try:
            # clear
            self.tree.delete(*self.tree.get_children())
            for w in self.news_frame.winfo_children():
                w.destroy()

            capital = float(self.capital_entry.get())
            tickers = [e.get().upper() for e in self.ticker_entries if e.get().strip()]
            if not tickers:
                messagebox.showwarning("Input Error","Enter ≥1 ticker"); return

            self.log(f"📊 Downloading data for {len(tickers)} tickers...")
            
            # download
            df = yf.download(tickers, period=self.period_combo.get(), interval=self.interval_combo.get())["Close"]
            if df.empty: 
                raise ValueError("No data fetched")

            # Handle single ticker case
            if len(tickers) == 1:
                df = df.to_frame(name=tickers[0])

            self.log(f"📈 Data shape: {df.shape}")
            
            # preprocess - clean data more carefully
            df = df.dropna()
            if len(df) < 20:
                raise ValueError("Insufficient data points for analysis")
            
            # Calculate returns
            returns = df.pct_change().dropna()
            
            # Remove extreme outliers (beyond 3 standard deviations)
            for col in returns.columns:
                mean_ret = returns[col].mean()
                std_ret = returns[col].std()
                returns[col] = returns[col].clip(lower=mean_ret - 3*std_ret, upper=mean_ret + 3*std_ret)
            
            # Add small noise to avoid singular covariance matrix
            np.random.seed(42)  # For reproducibility
            noise = np.random.normal(0, 1e-8, returns.shape)
            returns += noise

            # split
            split = max(10, int(0.7*len(returns)))  # Ensure minimum training data
            train, self.test_set = returns.iloc[:split], returns.iloc[split:]
            
            self.log(f"📊 Training set: {len(train)} samples, Test set: {len(self.test_set)} samples")

            # Check for numerical issues
            cov_matrix = train.cov()
            if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
                raise ValueError("Covariance matrix contains NaN or Inf values")
            
            # Check if covariance matrix is positive definite
            eigenvals = np.linalg.eigvals(cov_matrix)
            if np.any(eigenvals <= 0):
                self.log("⚠️ Covariance matrix is not positive definite, regularizing...")
                # Add small regularization
                cov_matrix += np.eye(len(cov_matrix)) * 1e-6

            # risk profile
            profile = self.profile_combo.get()
            params = RISK_PROFILES[profile]
            min_w, max_w = params["min_w"], params["max_w"]

            # optimize
            n = train.shape[1]
            
            # Better initial guess - equal weights
            x0 = np.full(n, 1/n)
            
            # Set up bounds and constraints
            bounds = Bounds(np.full(n, min_w), np.full(n, max_w))
            
            # Equality constraint: weights sum to 1
            constraint = LinearConstraint(np.ones((1, n)), [1], [1])

            self.log(f"⏳ Optimizing portfolio ({profile} profile)...")
            
            # Try multiple optimization methods
            methods = ['SLSQP', 'trust-constr']
            res = None
            
            for method in methods:
                try:
                    self.log(f"🔍 Trying {method} method...")
                    
                    if method == 'SLSQP':
                        res = minimize(
                            MV_criterion, x0, args=(train.values,), 
                            method=method,
                            bounds=bounds, 
                            constraints=constraint,
                            options={'maxiter': 1000, 'ftol': 1e-8, 'disp': False}
                        )
                    else:  # trust-constr
                        res = minimize(
                            MV_criterion, x0, args=(train.values,), 
                            method=method,
                            bounds=bounds, 
                            constraints=constraint,
                            options={'maxiter': 1000, 'gtol': 1e-6, 'disp': False}
                        )
                    
                    if res.success:
                        self.log(f"✅ {method} optimization succeeded!")
                        break
                    else:
                        self.log(f"⚠️ {method} failed: {res.message}")
                        
                except Exception as e:
                    self.log(f"❌ {method} error: {str(e)}")
                    continue
            
            # If optimization failed, use fallback methods
            if res is None or not res.success:
                self.log("🔄 Using fallback: Equal weights...")
                weights = np.full(n, 1/n)
                
                # Try simple risk parity as fallback
                try:
                    volatilities = np.sqrt(np.diag(cov_matrix))
                    inv_vol = 1 / volatilities
                    weights = inv_vol / np.sum(inv_vol)
                    
                    # Apply bounds
                    weights = np.clip(weights, min_w, max_w)
                    weights = weights / np.sum(weights)  # Renormalize
                    
                    self.log("✅ Applied risk parity weighting")
                except:
                    weights = np.full(n, 1/n)
                    self.log("⚠️ Using equal weights as final fallback")
            else:
                weights = res.x
                # Ensure weights sum to 1 (numerical precision)
                weights = weights / np.sum(weights)

            self.X_MV = weights
            self.log(f"🎯 Final weights: {[f'{w:.4f}' for w in weights]}")

            # Get live prices
            self.log("💰 Fetching current prices...")
            try:
                latest_data = yf.download(tickers, period="1d", interval="1d")["Close"]
                if len(tickers) == 1:
                    latest = {tickers[0]: latest_data.iloc[-1]}
                else:
                    latest = latest_data.iloc[-1].to_dict()
            except:
                # Fallback to last available price from historical data
                latest = df.iloc[-1].to_dict()
                self.log("⚠️ Using last historical price as current price")

            # Calculate metrics
            if len(self.test_set) > 0:
                portfolio_returns = np.dot(self.test_set, weights)
                ann_ret = np.mean(portfolio_returns) * 252  # Daily to annual
                ann_vol = np.std(portfolio_returns) * np.sqrt(252)
                sharpe = (ann_ret - 0.025) / ann_vol if ann_vol > 0 else 0
            else:
                ann_ret = ann_vol = sharpe = 0
                self.log("⚠️ No test data available for metrics calculation")

            # Populate results table
            for i, ticker in enumerate(df.columns):
                w = weights[i]
                price = latest.get(ticker, 0)
                allocation = w * capital
                self.tree.insert("", "end", values=(
                    ticker, 
                    f"₹{price:.2f}", 
                    f"{w:.4f}", 
                    f"₹{allocation:,.0f}"
                ))

            self.metrics_label.configure(
                text=f"Expected Return: {ann_ret:.2%} | Volatility: {ann_vol:.2%} | Sharpe Ratio: {sharpe:.2f}"
            )

            # Fetch news
            self.log("📰 Fetching news...")
            self.fetch_news(tickers)
            
            self.log("🎉 Portfolio optimization completed successfully!")

        except Exception as e:
            error_msg = f"❌ Error during optimization: {str(e)}"
            self.log(error_msg)
            messagebox.showerror("Optimization Error", str(e))

    def fetch_news(self, tickers):
        """Fetch news for given tickers"""
        for ticker in tickers:
            try:
                ticker_obj = yf.Ticker(ticker)
                news = ticker_obj.news[:3]  # Top 3 news items
                
                if news:
                    # Create header
                    header = ctk.CTkLabel(
                        self.news_frame, 
                        text=f"📈 {ticker} News:", 
                        font=("Arial", 12, "bold")
                    )
                    header.pack(anchor="w", padx=10, pady=(10, 5))
                    
                    # Add news items
                    for item in news:
                        title = item.get("title", "No title")
                        news_label = ctk.CTkLabel(
                            self.news_frame,
                            text=f"• {title}",
                            wraplength=600,
                            justify="left"
                        )
                        news_label.pack(anchor="w", padx=20, pady=2)
                        
            except Exception as e:
                self.log(f"⚠️ Could not fetch news for {ticker}: {str(e)}")

    def plot_cumulative_return(self):
        try:
            if self.X_MV is None:
                messagebox.showwarning("No Data", "Please optimize portfolio first")
                return
                
            # Clear previous plot
            canvas = getattr(self, 'canvas', None)
            if canvas: 
                canvas.get_tk_widget().destroy()

            # Calculate portfolio returns
            portfolio_returns = np.dot(self.test_set, self.X_MV)
            cumulative_returns = (1 + portfolio_returns).cumprod()
            
            # Get benchmark data
            benchmark_ticker = self.benchmark_options[self.benchmark_combo.get()]
            
            try:
                benchmark_data = yf.download(
                    benchmark_ticker, 
                    start=self.test_set.index[0], 
                    end=self.test_set.index[-1]
                )["Close"]
                
                benchmark_returns = benchmark_data.pct_change().dropna()
                
                # Align benchmark with portfolio dates
                benchmark_aligned = benchmark_returns.reindex(self.test_set.index, method='nearest')
                benchmark_aligned = benchmark_aligned.dropna()
                
                benchmark_cumulative = (1 + benchmark_aligned).cumprod()
                
            except Exception as e:
                self.log(f"⚠️ Could not fetch benchmark data: {str(e)}")
                benchmark_cumulative = None

            # Create plot
            fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
            
            # Plot portfolio
            ax.plot(
                cumulative_returns.index, 
                (cumulative_returns - 1) * 100, 
                label="Portfolio", 
                linewidth=2, 
                color='blue'
            )
            
            # Plot benchmark if available
            if benchmark_cumulative is not None:
                ax.plot(
                    benchmark_cumulative.index, 
                    (benchmark_cumulative - 1) * 100, 
                    label=self.benchmark_combo.get(), 
                    linestyle="--", 
                    color='red',
                    alpha=0.7
                )
            
            ax.set_title("Cumulative Returns Comparison", fontsize=14, fontweight='bold')
            ax.set_ylabel("Cumulative Return (%)")
            ax.set_xlabel("Date")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Format dates on x-axis
            fig.autofmt_xdate()
            fig.tight_layout()

            # Embed plot in tkinter
            self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
            
            self.log("📊 Cumulative returns plot generated successfully")
            
        except Exception as e:
            error_msg = f"❌ Plot error: {str(e)}"
            self.log(error_msg)
            messagebox.showerror("Plot Error", str(e))

    def export_to_csv(self):
        try:
            if not self.tree.get_children():
                messagebox.showwarning("No Data", "No portfolio data to export")
                return
                
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if not file_path:
                return
                
            # Extract data from tree
            data = []
            for item in self.tree.get_children():
                values = self.tree.item(item)["values"]
                data.append(values)
            
            # Create DataFrame
            df = pd.DataFrame(data, columns=["Ticker", "Price", "Weight", "Allocation"])
            
            # Add timestamp
            df["Export_Date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Save to CSV
            df.to_csv(file_path, index=False)
            
            self.log(f"💾 Portfolio exported to {file_path}")
            messagebox.showinfo("Export Successful", f"Data exported to {file_path}")
            
        except Exception as e:
            error_msg = f"❌ Export error: {str(e)}"
            self.log(error_msg)
            messagebox.showerror("Export Error", str(e))

    def log(self, message):
        """Add timestamped message to log"""
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        full_message = f"{timestamp} {message}\n"
        self.log_textbox.insert("end", full_message)
        self.log_textbox.see("end")


if __name__ == "__main__":
    app = PortfolioOptimizerApp()
    app.mainloop()