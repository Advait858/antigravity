import React from 'react';

interface RiskGaugeProps {
    score: number; // 1 to 10
}

export const RiskGauge: React.FC<RiskGaugeProps> = ({ score }) => {
    // Normalize 1-10 to 0-100%
    // 1 = Safe (Green), 10 = Risky (Red)
    const percentage = Math.min(100, Math.max(0, (score / 10) * 100));

    const getColor = (s: number) => {
        if (s < 4) return '#0ecb81'; // Green
        if (s < 7) return '#f0b90b'; // Yellow
        return '#f6465d'; // Red
    };

    const color = getColor(score);

    return (
        <div className="risk-gauge" style={{
            textAlign: 'center',
            padding: '10px',
            background: '#1e2329',
            borderRadius: '8px'
        }}>
            <div style={{
                fontSize: '10px',
                textTransform: 'uppercase',
                color: '#848e9c',
                marginBottom: '5px'
            }}>
                Risk Score
            </div>

            {/* Gauge Bar */}
            <div style={{
                height: '8px',
                width: '100%',
                background: '#2b3139',
                borderRadius: '4px',
                overflow: 'hidden',
                marginBottom: '5px'
            }}>
                <div style={{
                    height: '100%',
                    width: `${percentage}%`,
                    background: color,
                    transition: 'width 0.5s ease'
                }} />
            </div>

            <div style={{
                fontSize: '18px',
                fontWeight: 'bold',
                color: color
            }}>
                {score.toFixed(1)}/10
            </div>
        </div>
    );
};
