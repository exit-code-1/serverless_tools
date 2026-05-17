interface BannerProps {
  kind?: 'info' | 'warn' | 'error';
  children: React.ReactNode;
}

export default function StatusBanner({ kind = 'info', children }: BannerProps) {
  return <div className={`banner ${kind === 'info' ? '' : kind}`}>{children}</div>;
}
