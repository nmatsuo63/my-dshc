import plotly.offline as py
import plotly.graph_objs as go

try:
    from google.colab import output 
    output.enable_custom_widget_manager()
except:
    py.init_notebook_mode()


def graph3d(mx, my, mz, point_x=None, point_y=None, point_z=None, markersize=12):
    
    # Creating the plot
    lines = []
    line_marker = dict(color='#0066FF', width=2)
    for i, j, k in zip(mx, my, mz):
        lines.append(go.Scatter3d(x=i, y=j, z=k, mode='lines', line=line_marker))

    # add point
    if not point_x is None:
        lines.append(go.Scatter3d(
                                          x = [point_x],
                                          y = [point_y],
                                          z = [point_z],
#                                           name = "x",
                                          type = "scatter3d",
                                          marker=dict(
                                            color='rgb(127, 127, 127)',
                                            size=markersize
                                          )
                                        )
                       )

    layout = go.Layout(
        title='Wireframe Plot',
        scene=dict(
            xaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            yaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            zaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            )
#             annotations = [
#                 dict(
#                     x = x_,
#                     y = y_,
#                     z = z_,
#                     text = "Point 2",
#                     textangle = 0,
#                     ax = 75,
#                     ay = 0,
#                     font = dict(
#                       color = "black",
#                       size = 12
#                     ),
#                     arrowcolor = "black",
#                     arrowsize = 3,
#                     arrowwidth = 1,
#                     arrowhead = 1
#                 )]
        ),
        showlegend=False,

    )
    fig = go.Figure(data=lines, layout=layout)
    py.iplot(fig, filename='wireframe_plot')