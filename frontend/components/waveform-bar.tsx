import React, { useEffect, useRef } from 'react';

interface AudioVisualizerProps {
  isActive: boolean;
  className?: string;
}

export const AudioVisualizer: React.FC<AudioVisualizerProps> = ({ isActive, className = '' }) => {
  const [heights, setHeights] = React.useState<number[]>([20, 30, 40, 25, 20]);
  const raf = useRef<number | undefined>(undefined);

  useEffect(() => {
    let running = true;
    function animate() {
      if (!running) return;
      const now = Date.now() / 200;
      setHeights(
        Array.from({ length: 5 }, (_, i) => {
          if (!isActive) return 20;
          // Create varying heights for the shapes
          const baseHeight = [30, 45, 60, 35, 25][i];
          const variation = Math.sin(now + i * 0.8) * 15;
          return Math.max(15, baseHeight + variation);
        })
      );
      raf.current = requestAnimationFrame(animate);
    }
    animate();
    return () => {
      running = false;
      if (raf.current) cancelAnimationFrame(raf.current);
    };
  }, [isActive]);

  return (
    <div className={`flex items-end justify-center gap-3 ${className}`}>
      {/* Circle */}
      <div
        className={`transition-all duration-150 ${
          isActive ? 'shadow-[0_0_20px_4px_rgba(255,255,255,0.6)]' : ''
        }`}
        style={{
          width: `${heights[0]}px`,
          height: `${heights[0]}px`,
          borderRadius: '50%',
          background: isActive ? 'rgba(255, 255, 255, 0.9)' : 'rgba(255, 255, 255, 0.3)',
          opacity: isActive ? 1 : 0.5,
        }}
      />

      {/* Tall vertical oval */}
      <div
        className={`transition-all duration-150 ${
          isActive ? 'shadow-[0_0_20px_4px_rgba(255,255,255,0.6)]' : ''
        }`}
        style={{
          width: `${heights[1] * 0.4}px`,
          height: `${heights[1]}px`,
          borderRadius: '50%',
          background: isActive ? 'rgba(255, 255, 255, 0.9)' : 'rgba(255, 255, 255, 0.3)',
          opacity: isActive ? 1 : 0.5,
        }}
      />

      {/* Largest vertical oval */}
      <div
        className={`transition-all duration-150 ${
          isActive ? 'shadow-[0_0_20px_4px_rgba(255,255,255,0.6)]' : ''
        }`}
        style={{
          width: `${heights[2] * 0.5}px`,
          height: `${heights[2]}px`,
          borderRadius: '50%',
          background: isActive ? 'rgba(255, 255, 255, 0.9)' : 'rgba(255, 255, 255, 0.3)',
          opacity: isActive ? 1 : 0.5,
        }}
      />

      {/* Circle */}
      <div
        className={`transition-all duration-150 ${
          isActive ? 'shadow-[0_0_20px_4px_rgba(255,255,255,0.6)]' : ''
        }`}
        style={{
          width: `${heights[3]}px`,
          height: `${heights[3]}px`,
          borderRadius: '50%',
          background: isActive ? 'rgba(255, 255, 255, 0.9)' : 'rgba(255, 255, 255, 0.3)',
          opacity: isActive ? 1 : 0.5,
        }}
      />

      {/* Circle */}
      <div
        className={`transition-all duration-150 ${
          isActive ? 'shadow-[0_0_20px_4px_rgba(255,255,255,0.6)]' : ''
        }`}
        style={{
          width: `${heights[4]}px`,
          height: `${heights[4]}px`,
          borderRadius: '50%',
          background: isActive ? 'rgba(255, 255, 255, 0.9)' : 'rgba(255, 255, 255, 0.3)',
          opacity: isActive ? 1 : 0.5,
        }}
      />
    </div>
  );
};

// Keep the old component for backward compatibility
export const WaveformBar: React.FC<{
  isActive: boolean;
  color?: string;
  barCount?: number;
  height?: number;
  className?: string;
}> = ({ isActive, color = '#3b82f6', barCount = 12, height = 40, className = '' }) => {
  const [wave, setWave] = React.useState<number[]>(Array(barCount).fill(0));
  const raf = useRef<number | undefined>(undefined);

  useEffect(() => {
    let running = true;
    function animate() {
      if (!running) return;
      const now = Date.now() / 300;
      setWave(
        Array.from({ length: barCount }, (_, i) => {
          if (!isActive) return 8;
          // Sine wave for smooth animation
          return 10 + Math.abs(Math.sin(now + i * 0.5)) * (height - 10);
        })
      );
      raf.current = requestAnimationFrame(animate);
    }
    animate();
    return () => {
      running = false;
      if (raf.current) cancelAnimationFrame(raf.current);
    };
  }, [isActive, barCount, height]);

  return (
    <div className={`flex items-end gap-1 ${className}`} style={{ height: height + 8 }}>
      {wave.map((h, i) => (
        <div
          key={i}
          className={`w-2 rounded-full transition-all duration-150 ${isActive ? 'shadow-[0_0_16px_2px_rgba(59,130,246,0.5)]' : ''}`}
          style={{
            height: `${h}px`,
            background: isActive
              ? `linear-gradient(180deg, ${color} 60%, #60a5fa 100%)`
              : '#334155',
            opacity: isActive ? 0.95 : 0.3,
            transitionProperty: 'height, opacity, background',
          }}
        />
      ))}
    </div>
  );
};
