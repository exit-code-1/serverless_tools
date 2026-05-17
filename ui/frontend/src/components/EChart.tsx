import { useEffect, useRef } from 'react';
import * as echarts from 'echarts';

interface EChartProps {
  option: Record<string, unknown>;
  height?: number | string;
}

export default function EChart({ option, height = 280 }: EChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const chart = echarts.init(containerRef.current, 'dark');
    chartRef.current = chart;
    const handleResize = () => chart.resize();
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      chart.dispose();
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    chartRef.current?.setOption(option as echarts.EChartsOption, true);
  }, [option]);

  return <div ref={containerRef} style={{ width: '100%', height }} />;
}
