"""
pdf_reporter.py - PDF Report Generation

Creates professional PDF reports with figures and statistics
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import sys


class PDFReporter:
    """Generates PDF analysis reports"""
    
    @staticmethod
    def generate_report(filepath, selections, neutron_data, xray_data, histogram_data=None):
        """
        Generate comprehensive PDF report
        
        Args:
            filepath: Path to save PDF
            selections: List of Selection objects
            neutron_data: 2D neutron array
            xray_data: 2D xray array
            histogram_data: Optional (hist, n_edges, x_edges)
            
        Returns:
            str: Path to generated PDF
        """
        print(f"Generating PDF report...", file=sys.stderr)
        
        from pathlib import Path
        filepath = Path(filepath)
        if filepath.suffix != '.pdf':
            filepath = filepath.with_suffix('.pdf')
        
        with PdfPages(filepath) as pdf:
            # Page 1: Title and summary
            PDFReporter._page_title(pdf, selections)
            
            # Page 2: Histogram with ROIs
            if histogram_data is not None:
                PDFReporter._page_histogram(pdf, histogram_data, selections)
            
            # Page 3: Slice views
            PDFReporter._page_slices(pdf, neutron_data, xray_data, selections)
            
            # Page 4: Statistics
            PDFReporter._page_statistics(pdf, selections, neutron_data, xray_data)
            
            # Metadata
            d = pdf.infodict()
            d['Title'] = 'BiTS Analysis Report'
            d['Author'] = 'BiTS v14.0'
            d['CreationDate'] = datetime.now()
        
        print(f"Report saved: {filepath}", file=sys.stderr)
        return str(filepath)
    
    @staticmethod
    def _page_title(pdf, selections):
        """Title page"""
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.9, 'BiTS Analysis Report', ha='center', fontsize=24, fontweight='bold')
        ax.text(0.5, 0.85, 'Bivariate Tomography Segmentation', ha='center', fontsize=14)
        
        # Date
        ax.text(0.5, 0.78, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
               ha='center', fontsize=12)
        ax.text(0.5, 0.75, 'BiTS Version 14.0', ha='center', fontsize=10, style='italic')
        
        # Summary
        summary = f"\nAnalysis Summary\n{'='*60}\n\n"
        summary += f"Total Selections: {len(selections)}\n"
        
        clusters = [s for s in selections if s.cluster_id is not None]
        if clusters:
            summary += f"Clusters: {len(clusters)}\n"
        
        manual = len(selections) - len(clusters)
        if manual > 0:
            summary += f"Manual Selections: {manual}\n"
        
        summary += f"\nSelections:\n"
        for i, sel in enumerate(selections[:15], 1):
            summary += f"  {i}. {sel.name}\n"
        if len(selections) > 15:
            summary += f"  ... and {len(selections)-15} more\n"
        
        ax.text(0.1, 0.65, summary, ha='left', va='top', fontsize=10, family='monospace')
        
        pdf.savefig(fig)
        plt.close(fig)
    
    @staticmethod
    def _page_histogram(pdf, histogram_data, selections):
        """Histogram page with ROIs"""
        hist, n_edges, x_edges = histogram_data
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot histogram
        extent = [n_edges[0], n_edges[-1], x_edges[0], x_edges[-1]]
        im = ax.imshow(np.log10(hist.T + 1), origin='lower', extent=extent,
                      aspect='auto', cmap='hot')
        
        ax.set_xlabel('Neutron Intensity', fontsize=13)
        ax.set_ylabel('X-ray Intensity', fontsize=13)
        ax.set_title('2D Bivariate Histogram with Selection ROIs', fontsize=15, fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='log10(Count + 1)')
        
        # Overlay ROIs
        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab10')
        
        for i, sel in enumerate(selections[:10]):  # Max 10 for readability
            if sel.histogram_roi is not None:
                color = sel.color if sel.color else cmap(i/10)
                roi_closed = np.vstack([sel.histogram_roi, sel.histogram_roi[0]])
                ax.plot(roi_closed[:, 0], roi_closed[:, 1], '--', 
                       color=color, linewidth=2, label=sel.name, alpha=0.9)
        
        if len(selections) <= 10:
            ax.legend(loc='best', fontsize=9)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    
    @staticmethod
    def _page_slices(pdf, neutron_data, xray_data, selections):
        """Slice visualization page"""
        fig, axes = plt.subplots(2, 2, figsize=(11, 11))
        fig.suptitle('Slice Visualizations', fontsize=16, fontweight='bold')
        
        # Neutron
        axes[0, 0].imshow(neutron_data, cmap='gray')
        axes[0, 0].set_title('Neutron Image', fontsize=13)
        axes[0, 0].axis('off')
        
        # X-ray
        axes[0, 1].imshow(xray_data, cmap='gray')
        axes[0, 1].set_title('X-ray Image', fontsize=13)
        axes[0, 1].axis('off')
        
        # Neutron + masks
        axes[1, 0].imshow(neutron_data, cmap='gray')
        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab10')
        for i, sel in enumerate(selections[:8]):
            if sel.spatial_mask is not None:
                color = sel.color if sel.color else cmap(i/10)
                overlay = np.zeros((*sel.spatial_mask.shape, 4))
                overlay[sel.spatial_mask] = [color[0], color[1], color[2], 0.5]
                axes[1, 0].imshow(overlay)
        axes[1, 0].set_title('Neutron + Selection Masks', fontsize=13)
        axes[1, 0].axis('off')
        
        # X-ray + masks
        axes[1, 1].imshow(xray_data, cmap='gray')
        for i, sel in enumerate(selections[:8]):
            if sel.spatial_mask is not None:
                color = sel.color if sel.color else cmap(i/10)
                overlay = np.zeros((*sel.spatial_mask.shape, 4))
                overlay[sel.spatial_mask] = [color[0], color[1], color[2], 0.5]
                axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('X-ray + Selection Masks', fontsize=13)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    
    @staticmethod
    def _page_statistics(pdf, selections, neutron_data, xray_data):
        """Statistics table page"""
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        ax.text(0.5, 0.95, 'Selection Statistics', ha='center', fontsize=16, fontweight='bold')
        
        # Table data
        table_data = [['Selection', 'Pixels', 'Vol%', 'N_Mean', 'N_Std', 'X_Mean', 'X_Std']]
        
        total_pixels = neutron_data.size
        skipped = 0
        
        for sel in selections:
            if sel.spatial_mask is not None and np.sum(sel.spatial_mask) > 0:
                mask = sel.spatial_mask
                
                # Check dimension match
                if mask.shape != neutron_data.shape:
                    skipped += 1
                    continue
                
                n_pix = np.sum(mask)
                
                n_vals = neutron_data[mask]
                x_vals = xray_data[mask]
                
                table_data.append([
                    sel.name[:25],
                    f'{n_pix:,}',
                    f'{100*n_pix/total_pixels:.1f}',
                    f'{np.mean(n_vals):.0f}',
                    f'{np.std(n_vals):.0f}',
                    f'{np.mean(x_vals):.0f}',
                    f'{np.std(x_vals):.0f}'
                ])
        
        # Create table
        table = ax.table(cellText=table_data, cellLoc='center',
                        loc='upper center', bbox=[0.05, 0.1, 0.9, 0.8])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Add warning if some selections were skipped
        if skipped > 0:
            ax.text(0.5, 0.05, 
                   f'⚠ {skipped} selection(s) excluded due to dimension mismatch',
                   ha='center', fontsize=10, style='italic', color='red')
        
        pdf.savefig(fig)
        plt.close(fig)
