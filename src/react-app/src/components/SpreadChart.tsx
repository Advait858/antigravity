import React, { useEffect, useRef } from 'react';
import { createChart, IChartApi, ISeriesApi, ColorType } from 'lightweight-charts';

interface SpreadChartProps {
    data: { time: number; value: number }[];
    pairName: string;
}

export const SpreadChart: React.FC<SpreadChartProps> = ({ data, pairName }) => {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const seriesRef = useRef<ISeriesApi<'Line'> | null>(null);

    useEffect(() => {
        if (chartContainerRef.current) {
            const chart = createChart(chartContainerRef.current, {
                layout: {
                    background: { type: ColorType.Solid, color: '#0b0e11' },
                    textColor: '#848e9c'
                },
                grid: {
                    vertLines: { color: '#1e2329' },
                    horzLines: { color: '#1e2329' }
                },
                width: chartContainerRef.current.clientWidth,
                height: 300,
                timeScale: {
                    timeVisible: true,
                    borderColor: '#2b3139',
                },
                rightPriceScale: {
                    borderColor: '#2b3139',
                },
            });

            const lineSeries = chart.addLineSeries({
                color: '#f0b90b',
                lineWidth: 2,
                title: 'Spread',
            });

            // Add Bands (Static visual guides for now, ideally dynamic)
            // We can't easily add bands in lightweight charts without plugins or multiple series
            // For visualized "Buy/Sell" zones, we'll just add horizontal lines roughly if needed, 
            // or rely on the user knowing the Z-score context.

            chartRef.current = chart;
            seriesRef.current = lineSeries;

            const handleResize = () => {
                if (chartContainerRef.current) {
                    chart.applyOptions({ width: chartContainerRef.current.clientWidth });
                }
            };

            window.addEventListener('resize', handleResize);

            return () => {
                window.removeEventListener('resize', handleResize);
                chart.remove();
            };
        }
    }, []);

    useEffect(() => {
        if (seriesRef.current && data.length > 0) {
            seriesRef.current.setData(data);
            if (chartRef.current) {
                chartRef.current.timeScale().fitContent();
            }
        }
    }, [data]);

    return (
        <div style={{ position: 'relative' }}>
            <div style={{
                position: 'absolute',
                top: '10px',
                left: '10px',
                zIndex: 5,
                color: '#848e9c',
                fontSize: '12px'
            }}>
                {pairName || "Select a Pair"}
            </div>
            <div ref={chartContainerRef} style={{ width: '100%' }} />
        </div>
    );
};
