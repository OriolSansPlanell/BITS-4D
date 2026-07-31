"""
editable_roi_handler.py - Make histogram ROI vertices draggable

Enables interactive editing of polygon and rectangle ROIs in histogram space
"""

import numpy as np
from matplotlib.patches import Circle, Rectangle as MPLRectangle
from matplotlib.lines import Line2D
import sys


class EditableROIHandler:
    """
    Makes histogram ROI vertices draggable for fine-tuning
    """
    
    def __init__(self, ax, roi_manager, update_callback=None):
        """
        Initialize editable ROI handler
        
        Args:
            ax: Matplotlib axes for histogram
            roi_manager: ROIManager instance
            update_callback: Function to call when ROI is modified
        """
        self.ax = ax
        self.roi_manager = roi_manager
        self.update_callback = update_callback
        
        # State
        self.dragging_vertex = None  # Index of vertex being dragged
        self.dragging_edge = None  # Which edge (for rectangle)
        self.vertex_artists = []  # Circle artists for vertices
        self.edge_artists = []  # Line artists for edges
        
        # Interaction parameters
        self.vertex_size = 8  # Radius in points
        self.pick_tolerance = 10  # Pixels
        self.vertex_color = 'yellow'
        self.vertex_edge_color = 'red'
        self.edge_color = 'red'
        self.edge_width = 2
        
        # Enable/disable state
        self.enabled = False
        
        # Event connections
        self.cid_press = None
        self.cid_release = None
        self.cid_motion = None
        
    def enable(self):
        """Enable editable ROI"""
        if self.enabled:
            return
            
        print("EditableROIHandler: Enabling", file=sys.stderr)
        
        # Connect events
        self.cid_press = self.ax.figure.canvas.mpl_connect(
            'button_press_event', self._on_press
        )
        self.cid_release = self.ax.figure.canvas.mpl_connect(
            'button_release_event', self._on_release
        )
        self.cid_motion = self.ax.figure.canvas.mpl_connect(
            'motion_notify_event', self._on_motion
        )
        
        self.enabled = True
        
        # Draw editable vertices
        self.draw_editable_roi()
        
    def disable(self):
        """Disable editable ROI"""
        if not self.enabled:
            return
            
        print("EditableROIHandler: Disabling", file=sys.stderr)
        
        # Disconnect events
        try:
            if self.cid_press:
                self.ax.figure.canvas.mpl_disconnect(self.cid_press)
            if self.cid_release:
                self.ax.figure.canvas.mpl_disconnect(self.cid_release)
            if self.cid_motion:
                self.ax.figure.canvas.mpl_disconnect(self.cid_motion)
        except Exception as e:
            print(f"  Warning disconnecting events: {e}", file=sys.stderr)
        
        self.enabled = False
        
        # Clear vertex artists
        self._clear_artists()
        
        print("  EditableROIHandler disabled successfully", file=sys.stderr)
        
    def draw_editable_roi(self):
        """Draw ROI with editable vertices"""
        if not self.roi_manager.has_roi():
            return
            
        # Clear previous artists
        self._clear_artists()
        
        if self.roi_manager.roi_type == 'polygon':
            self._draw_editable_polygon()
        elif self.roi_manager.roi_type == 'rectangle':
            self._draw_editable_rectangle()
        
        # Redraw canvas
        self.ax.figure.canvas.draw_idle()
    
    def _draw_editable_polygon(self):
        """Draw polygon with draggable vertices"""
        points = self.roi_manager.polygon_points
        
        # Draw edges
        xs = np.append(points[:, 0], points[0, 0])
        ys = np.append(points[:, 1], points[0, 1])
        line = Line2D(xs, ys, color=self.edge_color, 
                     linewidth=self.edge_width, zorder=10)
        self.ax.add_line(line)
        self.edge_artists.append(line)
        
        # Draw draggable vertices
        for i, (x, y) in enumerate(points):
            circle = Circle((x, y), radius=self._data_to_display_units(self.vertex_size),
                          facecolor=self.vertex_color,
                          edgecolor=self.vertex_edge_color,
                          linewidth=2,
                          zorder=20,
                          picker=True)
            self.ax.add_patch(circle)
            self.vertex_artists.append(circle)
    
    def _draw_editable_rectangle(self):
        """Draw rectangle with draggable corners"""
        x_min, y_min, x_max, y_max = self.roi_manager.rectangle
        
        # Draw rectangle outline
        points = np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ])
        
        xs = np.append(points[:, 0], points[0, 0])
        ys = np.append(points[:, 1], points[0, 1])
        line = Line2D(xs, ys, color=self.edge_color,
                     linewidth=self.edge_width, zorder=10)
        self.ax.add_line(line)
        self.edge_artists.append(line)
        
        # Draw draggable corners
        for i, (x, y) in enumerate(points):
            circle = Circle((x, y), radius=self._data_to_display_units(self.vertex_size),
                          facecolor=self.vertex_color,
                          edgecolor=self.vertex_edge_color,
                          linewidth=2,
                          zorder=20,
                          picker=True)
            self.ax.add_patch(circle)
            self.vertex_artists.append(circle)
    
    def _clear_artists(self):
        """Clear all vertex and edge artists"""
        import sys
        
        # Remove vertex artists (Circles)
        for artist in self.vertex_artists:
            try:
                # Try standard remove
                artist.remove()
            except (NotImplementedError, ValueError, AttributeError) as e:
                # Fallback: remove from patches collection
                try:
                    if artist in self.ax.patches:
                        self.ax.patches.remove(artist)
                except Exception as e2:
                    print(f"  Warning: Could not remove vertex artist: {e2}", file=sys.stderr)
        
        # Remove edge artists (Lines)
        for artist in self.edge_artists:
            try:
                # Try standard remove
                artist.remove()
            except (NotImplementedError, ValueError, AttributeError) as e:
                # Fallback: remove from lines collection
                try:
                    if artist in self.ax.lines:
                        self.ax.lines.remove(artist)
                except Exception as e2:
                    print(f"  Warning: Could not remove edge artist: {e2}", file=sys.stderr)
        
        # Clear lists
        self.vertex_artists = []
        self.edge_artists = []
        
        # Force canvas redraw to remove visual artifacts
        try:
            self.ax.figure.canvas.draw_idle()
        except:
            pass
    
    def _data_to_display_units(self, pixels):
        """Convert pixel size to data units for vertex radius"""
        # Approximate conversion - this varies by zoom level
        xlim = self.ax.get_xlim()
        fig_width = self.ax.figure.get_figwidth() * self.ax.figure.dpi
        data_width = xlim[1] - xlim[0]
        return (pixels / fig_width) * data_width * 2  # Adjust multiplier as needed
    
    def _on_press(self, event):
        """Handle mouse press - check if clicking on vertex"""
        if not self.enabled or event.inaxes != self.ax:
            return
        
        if event.button != 1:  # Only left click
            return
        
        # Check if clicking on a vertex
        vertex_idx = self._find_nearest_vertex(event.xdata, event.ydata)
        if vertex_idx is not None:
            self.dragging_vertex = vertex_idx
            print(f"  Started dragging vertex {vertex_idx}", file=sys.stderr)
    
    def _on_motion(self, event):
        """Handle mouse motion - drag vertex"""
        if not self.enabled or self.dragging_vertex is None:
            return
        
        if event.inaxes != self.ax or event.xdata is None:
            return
        
        # Update vertex position
        if self.roi_manager.roi_type == 'polygon':
            self.roi_manager.polygon_points[self.dragging_vertex] = [
                event.xdata, event.ydata
            ]
        elif self.roi_manager.roi_type == 'rectangle':
            # Update rectangle by moving corner
            self._update_rectangle_corner(self.dragging_vertex, event.xdata, event.ydata)
        
        # Redraw
        self.draw_editable_roi()
        
        # Trigger update callback
        if self.update_callback:
            self.update_callback()
    
    def _on_release(self, event):
        """Handle mouse release"""
        if self.dragging_vertex is not None:
            print(f"  Finished dragging vertex {self.dragging_vertex}", file=sys.stderr)
        self.dragging_vertex = None
    
    def _find_nearest_vertex(self, x, y):
        """Find vertex nearest to click point"""
        if self.roi_manager.roi_type == 'polygon':
            points = self.roi_manager.polygon_points
        elif self.roi_manager.roi_type == 'rectangle':
            x_min, y_min, x_max, y_max = self.roi_manager.rectangle
            points = np.array([
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max]
            ])
        else:
            return None
        
        # Compute distances in data coordinates
        distances = np.sqrt((points[:, 0] - x)**2 + (points[:, 1] - y)**2)
        
        # Convert tolerance from pixels to data units
        tolerance = self._data_to_display_units(self.pick_tolerance)
        
        min_idx = np.argmin(distances)
        if distances[min_idx] < tolerance:
            return min_idx
        
        return None
    
    def _update_rectangle_corner(self, corner_idx, new_x, new_y):
        """Update rectangle by dragging a corner"""
        x_min, y_min, x_max, y_max = self.roi_manager.rectangle
        
        if corner_idx == 0:  # Bottom-left
            x_min, y_min = new_x, new_y
        elif corner_idx == 1:  # Bottom-right
            x_max, y_min = new_x, new_y
        elif corner_idx == 2:  # Top-right
            x_max, y_max = new_x, new_y
        elif corner_idx == 3:  # Top-left
            x_min, y_max = new_x, new_y
        
        # Ensure min < max
        if x_max < x_min:
            x_min, x_max = x_max, x_min
        if y_max < y_min:
            y_min, y_max = y_max, y_min
        
        # Update rectangle
        self.roi_manager.rectangle = (x_min, y_min, x_max, y_max)
