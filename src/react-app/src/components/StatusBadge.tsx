/**
 * StatusBadge Component
 * 
 * Queries get_health() from canister to verify data source.
 * Shows RED "INVALID SUBMISSION" if backend is simulated.
 * Shows GREEN if data is real/live.
 */

import { useEffect, useState } from 'react';
import { useCanister } from '../hooks/useCanister';

interface HealthData {
    status: string;
    version: string;
    price_source: string;
    assets_tracked: number;
    active_signals: number;
    heartbeat_count: number;
}

export function StatusBadge() {
    const { queryCanister, isConnected } = useCanister();
    const [health, setHealth] = useState<HealthData | null>(null);
    const [isValid, setIsValid] = useState<boolean | null>(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const checkHealth = async () => {
            try {
                const result = await queryCanister('get_health');
                if (result) {
                    const data = JSON.parse(result as string);
                    setHealth(data);

                    // Validation: Check if data is real
                    const isRealData =
                        data.price_source !== 'none' &&
                        data.price_source !== 'simulated' &&
                        data.price_source !== 'mock' &&
                        data.assets_tracked > 0;

                    setIsValid(isRealData);
                }
            } catch (err) {
                setError('Failed to connect to canister');
                setIsValid(false);
            }
        };

        checkHealth();
        const interval = setInterval(checkHealth, 60000); // Check every minute

        return () => clearInterval(interval);
    }, [queryCanister]);

    if (error) {
        return (
            <div className="status-badge invalid">
                <span className="status-dot red"></span>
                <span>OFFLINE</span>
            </div>
        );
    }

    if (isValid === null) {
        return (
            <div className="status-badge">
                <div className="spinner" style={{ width: 12, height: 12, marginRight: 4 }}></div>
                <span>Checking...</span>
            </div>
        );
    }

    if (!isValid) {
        return (
            <div className="status-badge invalid">
                <span className="status-dot red"></span>
                <span>INVALID SUBMISSION</span>
            </div>
        );
    }

    return (
        <div className="status-badge valid">
            <span className="status-dot green"></span>
            <span>LIVE ({health?.price_source?.toUpperCase()})</span>
        </div>
    );
}
