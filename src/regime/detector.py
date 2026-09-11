"""
K-Means market regime detector.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

REGIME_NAMES = {0: 'SIDEWAYS', 1: 'BULLISH', 2: 'BEARISH', 3: 'HIGH_VOL'}

class RegimeDetector:
    def __init__(self, n_regimes=4, rolling_window=20, seed=42):
        self.n_regimes = n_regimes
        self.rolling_window = rolling_window
        self.seed = seed
        self.kmeans = KMeans(n_clusters=n_regimes, random_state=seed, n_init=10)
        self.scaler = StandardScaler()
        self._is_fitted = False
        # Mapping from K-Means cluster ID to our canonical Regime ID
        self._cluster_to_regime = {}

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'Close' not in df.columns:
            return pd.DataFrame()
            
        close = df['Close']
        returns = close.pct_change()
        
        # Features for clustering
        feats = pd.DataFrame(index=df.index)
        feats['rolling_return'] = returns.rolling(self.rolling_window).sum()
        feats['rolling_vol'] = returns.rolling(self.rolling_window).std()
        
        if 'EMA_slope' in df.columns:
            feats['ema_slope'] = df['EMA_slope']
        else:
            ema5 = close.ewm(span=5, adjust=False).mean()
            feats['ema_slope'] = (ema5 - ema5.shift(5)) / ema5.shift(5).replace(0, np.nan)
            
        if 'ATR_pct' in df.columns:
            feats['atr_pct'] = df['ATR_pct']
        else:
            feats['atr_pct'] = 0.0 # Fallback
            
        if 'volume_change' in df.columns:
            feats['volume_change'] = df['volume_change']
        else:
            feats['volume_change'] = 0.0
            
        return feats.ffill().fillna(0)

    def fit(self, df: pd.DataFrame):
        feats = self._compute_features(df)
        if len(feats) < self.n_regimes:
            return self
            
        X = self.scaler.fit_transform(feats)
        labels = self.kmeans.fit_predict(X)
        
        feats['cluster'] = labels
        # Analyze clusters to assign canonical regime IDs
        # 0=SIDEWAYS, 1=BULLISH, 2=BEARISH, 3=HIGH_VOL
        cluster_stats = feats.groupby('cluster')[['rolling_return', 'rolling_vol']].mean()
        
        # Sort by return to find Bearish and Bullish
        sorted_by_ret = cluster_stats.sort_values('rolling_return')
        bearish_cluster = sorted_by_ret.index[0]
        bullish_cluster = sorted_by_ret.index[-1]
        
        # Of the remaining two, higher vol is HIGH_VOL, lower vol is SIDEWAYS
        remaining = [c for c in cluster_stats.index if c not in [bearish_cluster, bullish_cluster]]
        if len(remaining) >= 2:
            if cluster_stats.loc[remaining[0], 'rolling_vol'] > cluster_stats.loc[remaining[1], 'rolling_vol']:
                highvol_cluster, sideways_cluster = remaining[0], remaining[1]
            else:
                highvol_cluster, sideways_cluster = remaining[1], remaining[0]
        else:
            highvol_cluster, sideways_cluster = -1, -1 # Fallback if n_regimes < 4

        self._cluster_to_regime[bearish_cluster] = 2
        self._cluster_to_regime[bullish_cluster] = 1
        if highvol_cluster != -1: self._cluster_to_regime[highvol_cluster] = 3
        if sideways_cluster != -1: self._cluster_to_regime[sideways_cluster] = 0
        
        self._is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise ValueError("RegimeDetector is not fitted yet.")
            
        df = df.copy()
        if len(df) < 2:
            # Fallback for very small df
            df['regime'] = 0
            df['regime_name'] = REGIME_NAMES[0]
            df['regime_confidence'] = 1.0
            for i in range(4): df[f'regime_{i}'] = 1.0 if i == 0 else 0.0
            return df
            
        feats = self._compute_features(df)
        X = self.scaler.transform(feats)
        
        cluster_ids = self.kmeans.predict(X)
        regime_ids = [self._cluster_to_regime.get(cid, 0) for cid in cluster_ids]
        
        df['regime'] = regime_ids
        df['regime_name'] = [REGIME_NAMES[rid] for rid in regime_ids]
        
        # Confidence based on distance to cluster center
        distances = self.kmeans.transform(X)
        min_dists = np.min(distances, axis=1)
        max_dists = np.max(distances, axis=1)
        range_dists = np.where(max_dists == min_dists, 1, max_dists - min_dists)
        df['regime_confidence'] = 1.0 - (min_dists / range_dists)
        df['regime_confidence'] = df['regime_confidence'].clip(0, 1)
        
        # One-hot encoding
        for i in range(4):
            df[f'regime_{i}'] = (df['regime'] == i).astype(float)
            
        return df

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.predict(df)

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'kmeans': self.kmeans,
                'scaler': self.scaler,
                'is_fitted': self._is_fitted,
                'cluster_to_regime': self._cluster_to_regime
            }, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.kmeans = data['kmeans']
            self.scaler = data['scaler']
            self._is_fitted = data['is_fitted']
            self._cluster_to_regime = data['cluster_to_regime']

