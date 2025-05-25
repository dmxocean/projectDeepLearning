"""
Visualization configuration for consistent output across all plotting functions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# Set default style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Figure sizes (width, height) in inches
FIGURE_SIZES = {
    'small': (6, 4),
    'medium': (8, 6),
    'large': (12, 8),
    'square': (8, 8),
    'wide': (14, 6),
    'tall': (6, 10),
    'attention': (10, 8),  # For attention heatmaps
    'comparison': (12, 10),  # For model comparison plots
    'grid': (15, 12),  # For multi-panel grids
}

# DPI settings
DPI_SETTINGS = {
    'screen': 100,
    'print': 300,
    'web': 150,
    'default': 100,
}

# Font configurations
FONT_SIZES = {
    'title': 16,
    'subtitle': 14,
    'label': 12,
    'tick': 10,
    'legend': 11,
    'annotation': 9,
}

FONT_FAMILIES = {
    'default': 'sans-serif',
    'serif': 'serif',
    'monospace': 'monospace',
}

# Color palettes
COLOR_PALETTES = {
    'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
    'sequential': plt.cm.viridis,
    'diverging': plt.cm.RdBu_r,
    'categorical': sns.color_palette("Set2"),
    'attention': plt.cm.hot,  # For attention heatmaps
    'comparison': sns.color_palette("husl", 8),  # For comparing models
    'error': ['#d62728', '#ff7f0e'],  # Error/warning colors
    'success': ['#2ca02c', '#1f77b4'],  # Success/info colors
}

# Line styles
LINE_STYLES = {
    'solid': '-',
    'dashed': '--',
    'dotted': ':',
    'dashdot': '-.',
}

# Marker styles
MARKER_STYLES = {
    'circle': 'o',
    'square': 's',
    'triangle': '^',
    'diamond': 'D',
    'star': '*',
    'plus': '+',
    'cross': 'x',
}

# Layout parameters
LAYOUT_PARAMS = {
    'tight_layout': True,
    'constrained_layout': False,
    'pad': 0.1,
    'h_pad': 1.0,
    'w_pad': 1.0,
    'rect': [0, 0, 1, 0.96],  # Leave space for suptitle
}

# Axis parameters
AXIS_PARAMS = {
    'grid': True,
    'grid_alpha': 0.3,
    'grid_linestyle': ':',
    'spine_width': 1.5,
    'tick_length': 5,
    'tick_width': 1.0,
}

# Legend parameters
LEGEND_PARAMS = {
    'loc': 'best',
    'frameon': True,
    'fancybox': True,
    'shadow': False,
    'framealpha': 0.9,
    'edgecolor': 'gray',
    'facecolor': 'white',
}

# Subplot parameters
SUBPLOT_PARAMS = {
    'hspace': 0.3,
    'wspace': 0.3,
    'left': 0.1,
    'right': 0.9,
    'top': 0.9,
    'bottom': 0.1,
}

# Animation parameters
ANIMATION_PARAMS = {
    'fps': 10,
    'interval': 100,  # milliseconds
    'repeat': True,
    'blit': True,
}

# Export parameters
EXPORT_PARAMS = {
    'bbox_inches': 'tight',
    'pad_inches': 0.1,
    'transparent': False,
    'facecolor': 'white',
    'edgecolor': 'none',
}


def get_figure_params(fig_type: str = 'default', dpi_type: str = 'default') -> Dict:
    """
    Get standardized figure parameters.
    
    Args:
        fig_type: Type of figure size ('small', 'medium', 'large', etc.)
        dpi_type: Type of DPI setting ('screen', 'print', 'web', 'default')
    
    Returns:
        Dictionary with figure parameters
    """
    return {
        'figsize': FIGURE_SIZES.get(fig_type, FIGURE_SIZES['medium']),
        'dpi': DPI_SETTINGS.get(dpi_type, DPI_SETTINGS['default']),
        'facecolor': 'white',
        'edgecolor': 'none',
    }


def apply_default_style():
    """Apply default styling to matplotlib."""
    # Set font sizes
    plt.rcParams['font.size'] = FONT_SIZES['tick']
    plt.rcParams['axes.titlesize'] = FONT_SIZES['title']
    plt.rcParams['axes.labelsize'] = FONT_SIZES['label']
    plt.rcParams['xtick.labelsize'] = FONT_SIZES['tick']
    plt.rcParams['ytick.labelsize'] = FONT_SIZES['tick']
    plt.rcParams['legend.fontsize'] = FONT_SIZES['legend']
    
    # Set other parameters
    plt.rcParams['figure.dpi'] = DPI_SETTINGS['default']
    plt.rcParams['savefig.dpi'] = DPI_SETTINGS['print']
    plt.rcParams['font.family'] = FONT_FAMILIES['default']
    
    # Grid settings
    plt.rcParams['axes.grid'] = AXIS_PARAMS['grid']
    plt.rcParams['grid.alpha'] = AXIS_PARAMS['grid_alpha']
    plt.rcParams['grid.linestyle'] = AXIS_PARAMS['grid_linestyle']
    
    # Layout
    plt.rcParams['figure.autolayout'] = LAYOUT_PARAMS['tight_layout']


def get_color_palette(palette_name: str = 'default', n_colors: int = None) -> List:
    """
    Get a color palette.
    
    Args:
        palette_name: Name of the palette
        n_colors: Number of colors to return (if applicable)
    
    Returns:
        List of colors or colormap
    """
    palette = COLOR_PALETTES.get(palette_name, COLOR_PALETTES['default'])
    
    if isinstance(palette, list) and n_colors:
        # Cycle through colors if needed
        return [palette[i % len(palette)] for i in range(n_colors)]
    
    return palette


def format_axis(ax, title: str = None, xlabel: str = None, ylabel: str = None,
                xlim: Tuple = None, ylim: Tuple = None, grid: bool = True):
    """
    Apply standard formatting to an axis.
    
    Args:
        ax: Matplotlib axis object
        title: Axis title
        xlabel: X-axis label
        ylabel: Y-axis label
        xlim: X-axis limits
        ylim: Y-axis limits
        grid: Whether to show grid
    """
    if title:
        ax.set_title(title, fontsize=FONT_SIZES['title'], pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_SIZES['label'])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_SIZES['label'])
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    ax.grid(grid, alpha=AXIS_PARAMS['grid_alpha'], 
            linestyle=AXIS_PARAMS['grid_linestyle'])
    
    # Set spine properties
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_PARAMS['spine_width'])
    
    # Set tick properties
    ax.tick_params(length=AXIS_PARAMS['tick_length'],
                   width=AXIS_PARAMS['tick_width'],
                   labelsize=FONT_SIZES['tick'])


def save_figure(fig, filename: str, dpi_type: str = 'print', **kwargs):
    """
    Save figure with standardized parameters.
    
    Args:
        fig: Matplotlib figure object
        filename: Output filename
        dpi_type: DPI setting type
        **kwargs: Additional savefig parameters
    """
    save_params = EXPORT_PARAMS.copy()
    save_params['dpi'] = DPI_SETTINGS.get(dpi_type, DPI_SETTINGS['print'])
    save_params.update(kwargs)
    
    fig.savefig(filename, **save_params)


# Apply default style on import
apply_default_style()