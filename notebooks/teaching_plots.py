# Copied from https://github.com/sods/gpss/blob/master/slides/teaching_plots.py

import matplotlib.pyplot as plt
import numpy as np
import mlai

def matrix(A, ax=None,
                bracket_width=3,
                bracket_style='square',
                type='values',
                colormap=False,
                highlight=False,
                highlight_row=None,
                highlight_col=None,
                highlight_width=3,
                highlight_color=[0,0,0],
                zoom=False,
                zoom_row=None,
                zoom_col=None,
                bracket_color=[0,0,0]):
    """Plot a matrix for visualisation in a slide or piece of text."""
    
    if ax is None:
        ax = plt.gca()
        
    nrows, ncols = A.shape
    
  
    x_lim = np.array([-0.75, ncols-0.25])
    y_lim = np.array([-0.75, nrows-0.25])
  
    ax.cla()
    handle=[]
    if type == 'image':
        handle =  ax.matshow(A)
    elif type == 'imagesc':
        handle =  ax.images(A, [np.array([A.min(), 0]).min(), A.max()])
    elif type == 'values':
        for i in range(nrows):
            for j in range(ncols):
                handle.append(ax.text(j, i, str(A[i, j]), horizontalalignment='center'))
    elif type == 'entries':
        for i in range(nrows):
            for j in range(ncols):
                if isstr(A[i,j]):
                    handle.append(ax.text(j, i, A[i, j], horizontalalignment='center'))
                    
                else:  
                    handle.append(ax.text(j+1, i+1, ' ', horizontalalignment='center'))
    elif type == 'patch':
        for i in range(nrows):
            for j in range(ncols):
                handle.append(ax.add_patch(
                    plt.Rectangle([i-0.5, j-0.5],
                                  width=1., height=1.,
                                  color=(A[i, j])*np.array([1, 1, 1]))))
    elif type == 'colorpatch':
        for i in range(nrows):
            for j in range(ncols):
                handle.append(ax.add_patch(
                    plt.Rectangle([i-0.5, j-0.5],
                                  width=1., height=1.,
                                  color=np.array([A[i, j, 0],
                                                  A[i, j, 1],
                                                  A[i, j, 2]]))))
                
                
    if bracket_style == 'boxes':
        x_lim = np.array([-0.5, ncols-0.5])
        ax.set_xlim(x_lim)
        y_lim = np.array([-0.5, nrows-0.5])
        ax.set_ylim(y_lim)
        for i in range(nrows+1):
            ax.add_line(plt.axhline(y=i-.5, #xmin=-0.5, xmax=ncols-0.5, 
                 color=bracket_color))
        for j in range(ncols+1):
            ax.add_line(plt.axvline(x=j-.5, #ymin=-0.5, ymax=nrows-0.5, 
                 color=bracket_color))
    elif bracket_style == 'square':
        tick_length = 0.25
        ax.plot([x_lim[0]+tick_length,
                     x_lim[0], x_lim[0],
                     x_lim[0]+tick_length],
                    [y_lim[0], y_lim[0],
                     y_lim[1], y_lim[1]],
                    linewidth=bracket_width,
                    color=np.array(bracket_color))
        ax.plot([x_lim[1]-tick_length, x_lim[1],
                              x_lim[1], x_lim[1]-tick_length],
                             [y_lim[0], y_lim[0], y_lim[1],
                              y_lim[1]],
                             linewidth=bracket_width, color=np.array(bracket_color))
      
    if highlight:       
        h_row = highlight_row
        h_col = highlight_col
        if isinstance(h_row, str) and h_row == ':':
            h_row = [0, nrows]
        if isinstance(h_col, str) and h_col == ':':
            h_col = [0, ncols]
        if len(h_row) == 1:
            h_row = [h_row, h_row]
        if len(h_col) == 1:
            h_col = [h_col, h_col]
        h_col.sort()
        h_row.sort()
        ax.add_line(plt.Line2D([h_col[0]-0.5, h_col[0]-0.5,
                              h_col[1]+0.5, h_col[1]+0.5,
                              h_col[0]-0.5],
                             [h_row[0]-0.5, h_row[1]+0.5,
                              h_row[1]+0.5, h_row[0]-0.5,
                              h_row[0]-0.5], color=highlight_color,
                               linewidth=highlight_width))
                    
    if zoom:      
        z_row = zoom_row
        z_col = zoom_col
        if isinstance(z_row, str) and z_row == ':':
            z_row = [1, nrows]
        if isinstance(z_col, str) and z_col == ':':
            z_col = [1, ncols]
        if len(z_row) == 1:
            z_row = [z_row, z_row]
        if len(z_col) == 1:
            z_col = [z_col, z_col]
        z_col.sort()
        z_row.sort()
        x_lim = [z_col[0]-0.5, z_col[1]+0.5]
        y_lim = [z_row[0]-0.5, z_row[1]+0.5]

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect('equal')
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis() #axis ij, axis equal, axis off

    if colormap:
        plt.colormap(obj=options.colormap) 
             
    return handle 


def base_plot(K, ind=[0, 1], ax=None,
              contour_color=[0., 0., 1],
              contour_style='-',
              contour_size=4,
              contour_markersize=4,
              contour_marker='x',
              fontsize=20):
    """
    % BASEPLOT Plot the contour of the covariance.
    % FORMAT
    % DESC creates the basic plot.
    % """

    blackcolor = [0,0,0]
    if ax is None:
        ax = plt.gca()
    v, U = np.linalg.eig(K[ind][:, ind])
    r = np.sqrt(v)
    theta = np.linspace(0, 2*np.pi, 200)[:, None]
    xy = np.dot(np.concatenate([r[0]*np.sin(theta), r[1]*np.cos(theta)], axis=1),U.T)
    cont = plt.Line2D(xy[:, 0], xy[:, 1],
                      linewidth=contour_size,
                      linestyle=contour_style,
                      color=contour_color)
    cent = plt.Line2D([0.], [0.],
                      marker=contour_marker,
                      color=contour_color,
                      linewidth=contour_size,
                      markersize=contour_markersize)

    ax.add_line(cont)
    ax.add_line(cent)

    thandle = []
    thandle.append(ax.set_xlabel('$f_{' + str(ind[1]+1)+ '}$',
                   fontsize=fontsize))
    thandle.append(ax.set_ylabel('$f_{' + str(ind[0]+1)+ '}$',
                   fontsize=fontsize))
    
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    x_lim = [-1.5, 1.5]
    y_lim = [-1.5, 1.5]
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    
    ax.add_line(plt.Line2D(x_lim, [0, 0], color=blackcolor))
    ax.add_line(plt.Line2D([0, 0], y_lim, color=blackcolor))

    ax.set_aspect('equal')
    #zeroAxes(gca, 0.025, 18, 'times')
    
    return cont, thandle, cent 

def two_point_pred(K, f, x, ax=None, ind=[0, 1],
                        conditional_linestyle = '-',
                        conditional_linecolor = [1., 0., 0.],
                        conditional_size = 4,
                        fixed_linestyle = '-',
                        fixed_linecolor = [0., 1., 0.],
                        fixed_size = 4,stub=None, start=0):
    
    subK = K[ind][:, ind]
    f = f[ind]
    x = x[ind]

    if ax is None:
        ax = plt.gca()

    cont, t, cent = base_plot(K, ind, ax=ax)
    if stub is not None:
        plt.savefig('./diagrams/' + stub + str(start) + '.svg')

    x_lim = ax.get_xlim()
    cont2 = plt.Line2D([x_lim[0], x_lim[1]], [f[0], f[0]], linewidth=fixed_size, linestyle=fixed_linestyle, color=fixed_linecolor)
    ax.add_line(cont2)

    if stub is not None:
        plt.savefig('./diagrams/' + stub + str(start+1) + '.svg')

    # # Compute conditional mean and variance
    f2_mean = subK[0, 1]/subK[0, 0]*f[0]
    f2_var = subK[1, 1] - subK[0, 1]/subK[0, 0]*subK[0, 1]
    x_val = np.linspace(x_lim[0], x_lim[1], 200)
    pdf_val = 1/np.sqrt(2*np.pi*f2_var)*np.exp(-0.5*(x_val-f2_mean)*(x_val-f2_mean)/f2_var)
    pdf = plt.Line2D(x_val, pdf_val+f[0], linewidth=conditional_size, linestyle=conditional_linestyle, color=conditional_linecolor)
    ax.add_line(pdf)
    if stub is not None:
        plt.savefig('./diagrams/' + stub + str(start+2) + '.svg')
    
    obs = plt.Line2D([f[1]], [f[0]], linewidth=10, markersize=10, color=fixed_linecolor, marker='o')
    ax.add_line(obs)
    if stub is not None:
        plt.savefig('./diagrams/' + stub + str(start+3) + '.svg')
    
    # load gpdistfunc

    #printLatexText(['\mappingFunction_1=' numsf2str(f[0], 3)], 'inputValueF1.tex', '../../../gp/tex/diagrams')


def kern_circular_sample(K, mu=None, filename=None, fig=None, num_samps=5, num_theta=200):

    """Make an animation of a circular sample from a covariance funciton."""

    tau = 2*np.pi
    n = K.shape[0]


    R1 = np.random.normal(size=(n, num_samps))
    U1 = np.dot(R1,np.diag(1/np.sqrt(np.sum(R1*R1, axis=0))))
    R2 = np.random.normal(size=(n, num_samps))
    R2 = R2 - np.dot(U1,np.diag(np.sum(R2*U1, axis=0)))
    R2 = np.dot(R2,np.diag(np.sqrt(np.sum(R1*R1, axis=0))/np.sqrt(np.sum(R2*R2, axis=0))))
    L = np.linalg.cholesky(K+np.diag(np.ones((n)))*1e-6)


    from matplotlib import animation
    x_lim = (0, 1)
    y_lim = (-2, 2)
    
    if fig is None:
        fig, _ = plt.subplots(figsize=(7,7))
    rect = 0, 0, 1., 1.
    ax = fig.add_axes(rect)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    line = []
    for i in range(num_samps):
        l, = ax.plot([], [], lw=2)
        line.append(l)
        
    # initialization function: plot the background of each frame
    def init():
        for i in range(num_samps):
            line[i].set_data([], [])
        return line,

    # animation function.  This is called sequentially
    def animate(i):
        theta = float(i)/num_theta*tau
        xc = np.cos(theta)
        yc = np.sin(theta)
        # generate 2d basis in t-d space
        coord = xc*R1 + yc*R2
        y = np.dot(L,coord)
        if mu is not None:
            y = y + mu
        x = np.linspace(0, 1, n)
        for i in range(num_samps):
            line[i].set_data(x, y[:, i])
        return line,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=num_theta, blit=True)
    if filename is not None:
        anim.save('./diagrams/' + filename, writer='imagemagick', fps=30)


def covariance_func(x, kernel_function,formula, shortname=None, longname=None, **args):
    """Write a slide on a given covariance matrix."""
    fig, ax = plt.subplots(figsize=((5,5)))
    hcolor = [1., 0., 1.]
    K = kernel_function(x, x, **args)
    obj = matrix(K, ax=ax, type='image', bracket_style='boxes')

    if shortname is not None:
        filename = shortname + '_covariance'
    else:
        filename = 'covariance'
    plt.savefig('./diagrams/' + filename + '.svg')

    ax.cla()
    kern_circular_sample(K, fig=fig, filename=filename + '.gif')

    out = '<h2>' + longname + ' Covariance</h2>'
    out += '\n\n'
    out += '<p><center>' + formula + '</center></p>'
    out += '<table>\n  <tr><td><img src="./diagrams/' +filename + '.svg"></td><td><img src="./diagrams/' + filename + '.gif"></td></tr>\n</table>'

    return out


def gaussian_height():
    h = np.linspace(0, 2.5, 1000)
    sigma2 = 0.0225
    mu = 1.7
    p = 1./np.sqrt(2*np.pi*sigma2)*np.exp(-(h-mu)**2/(2*sigma2**2))
    f2, ax2 = plt.subplots(figsize=(7, 3.5))
    ax2.plot(h, p, 'b-', linewidth=3)
    ylim = (0, 3)
    ax2.vlines(mu, ylim[0], ylim[1], colors='r', linewidth=3)
    ax2.set_ylim(ylim)
    ax2.set_xlim(1.4, 2.0)
    ax2.set_xlabel('$h/m$', fontsize=20)
    ax2.set_ylabel('$p(h|\mu, \sigma^2)$', fontsize = 20)
    f2.savefig('./diagrams/gaussian_of_height.svg')

def under_determined_system():
    """Visualise what happens in an under determined system with linear regression."""
    x = 1.
    y = 3.
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(x, y, 'o', markersize=10, linewidth=3, color=[1., 0., 0.])
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ylim = [0, 5]
    xlim = [0, 3]
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_xlabel('$x$', fontsize=20)
    ax.set_ylabel('$y$', fontsize=20)
    fig.savefig('./diagrams/one_point0.svg')

    xvals = np.linspace(0, 3, 2)[:, None]
    count=0
    for i in range(100):
        c = np.random.normal(size=(1,1))*2
        m = (y - c)/x
        yvals = m*xvals+c
        ax.plot(xvals, yvals, '-', linewidth=2, color=[0., 0., 1.])
        if i < 9 or i == 100:
            count += 1
            fig.savefig('./diagrams/one_point' + str(count) + '.svg')


def bayes_update():
    "Visualise the updating of a posterior of Bayesian inference for a Gaussian lieklihood."""
    fig, ax = plt.subplots(figsize=(7,7))
    num_points = 1000
    x_max = 4
    x_min = -3

    y = np.array([[1.]])
    prior_mean = np.array([[0.]])
    prior_var = np.array([[.1]])

    noise = mlai.Gaussian(offset=np.array([0.6]), scale=np.array(np.sqrt(0.05)))


    f = np.linspace(x_min, x_max, num_points)[:, None]
    ln_prior_curve = -0.5*(np.log(2*np.pi*prior_var) + (f-prior_mean)*(f-prior_mean)/prior_var)
    ln_likelihood_curve = np.zeros(ln_prior_curve.shape)
    for i in range(num_points):
        ln_likelihood_curve[i] = noise.log_likelihood(f[i][None, :], 
                                                      np.array([[np.finfo(float).eps]]), 
                                                      y)
    ln_marginal_likelihood = noise.log_likelihood(prior_mean, prior_var, y);

    prior_curve = np.exp(ln_prior_curve) 
    likelihood_curve = np.exp(ln_likelihood_curve)
    marginal_curve = np.exp(ln_marginal_likelihood)

    ln_posterior_curve = ln_likelihood_curve + ln_prior_curve - ln_marginal_likelihood
    posterior_curve = np.exp(ln_posterior_curve)

    g, dlnZ_dvs = noise.grad_vals(prior_mean, prior_var, y)

    nu = g*g - 2*dlnZ_dvs

    approx_var = prior_var - prior_var*prior_var*nu
    approx_mean = prior_mean + prior_var*g

    ln_approx_curve = -0.5*np.log(2*np.pi*approx_var)-0.5*(f-approx_mean)*(f-approx_mean)/approx_var

    approx_curve = np.exp(ln_approx_curve)
    noise
    xlim = [x_min, x_max] 
    ylim = [0, np.vstack([approx_curve, likelihood_curve, 
                          posterior_curve, prior_curve]).max()*1.1]

    fig, ax = plt.subplots(figsize=(7,7))

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_yticks([0, 1, 2, 3, 4, 5])

    ax.vlines(xlim[0], ylim[0], ylim[1], color=[0., 0., 0.]) 
    ax.hlines(ylim[0], xlim[0], xlim[1], color=[0., 0., 0.]) 

    ax.plot(f, prior_curve, color=[1, 0., 0.], linewidth=3)
    ax.text(3.5, 2, '$p(c) = \mathcal{N}(c|0, \\alpha_1)$', horizontalalignment='center') 
    plt.savefig('./diagrams/dem_gaussian1.svg')

    ax.plot(f, likelihood_curve, color=[0, 0, 1], linewidth=3)
    ax.text(3.5, 1.5,'$p(y|m, c, x, \\sigma^2)=\mathcal{N}(y|mx+c,\\sigma^2)$', horizontalalignment='center') 
    plt.savefig('./diagrams/dem_gaussian2.svg')

    ax.plot(f, posterior_curve, color=[1, 0, 1], linewidth=3)
    ax.text(3.5, 1, '$p(c|y, m, x, \\sigma^2)=$', horizontalalignment='center') 
    plt.text(3.5, 0.75, '$\mathcal{N}\\left(c|\\frac{y-mx}{1+\\sigma^2\\alpha_1},(\\sigma^{-2}+\\alpha_1^{-1})^{-1}\\right)$', horizontalalignment='center') 
    plt.savefig('./diagrams/dem_gaussian3.svg')

def dem_two_point_sample(kernel_function, **args):
    """Make plots for the two data point sample example for explaining gaussian processes."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=((10,5)))
    hcolor = [1., 0., 1.]
    x = np.linspace(-1, 1, 25)[:, None]
    K = kernel_function(x, x, **args)
    obj = matrix(K, ax=ax[1], type='image')
    ax[1].set_xlabel('$i$',fontsize=16)
    ax[1].set_ylabel('$i^\prime$',fontsize=16)
    #fig.colorbar(mappable=obj, ax=ax[1])
    #ax[1].set_axis('off')
    plt.savefig('./diagrams/dem_two_point_sample0.svg')

    f = np.random.multivariate_normal(np.zeros(25), K, size=1)
    ax[0].plot(range(1, 26), f.flatten(), 'o', markersize=5, linewidth=3, color=[1., 0., 0.])
    ax[0].set_xticks(range(1, 26, 2))
    ax[0].set_yticks([-2, -1, 0, 1, 2])
    ylim = [-2, 2]
    xlim = [0, 26]
    ax[0].set_ylim(ylim)
    ax[0].set_xlim(xlim)
    ax[0].set_xlabel('$i$', fontsize=20)
    ax[0].set_ylabel('$f$', fontsize=20)
    plt.savefig('./diagrams/dem_two_point_sample1.svg')

    ax[0].plot(np.array([1, 2]), [f[0,0], f[0,1]], 'o', markersize=10, linewidth=5, color=hcolor)
    plt.savefig('./diagrams/dem_two_point_sample2.svg')
    #plt.Circle?

    obj = matrix(K, ax=ax[1], type='image', 
                      highlight=True, 
                      highlight_row=[0, 1], 
                      highlight_col=[0,1], 
                      highlight_color=hcolor)
    ax[1].set_xlabel('$i$',fontsize=16)
    ax[1].set_ylabel('$i^\prime$',fontsize=16)
    plt.savefig('./diagrams/dem_two_point_sample3.svg')

    obj = matrix(K, ax=ax[1], type='image', 
                      highlight=True, 
                      highlight_row=[0, 1], 
                      highlight_col=[0,1], 
                      highlight_color=hcolor,
                      highlight_width=5,
                     zoom=True,
                     zoom_row=[0, 9],
                     zoom_col=[0, 9])
    ax[1].set_xlabel('$i$',fontsize=16)
    ax[1].set_ylabel('$i^\prime$',fontsize=16)
    plt.savefig('./diagrams/dem_two_point_sample4.svg')

    obj = matrix(K, ax=ax[1], type='image', 
                      highlight=True, 
                      highlight_row=[0, 1], 
                      highlight_col=[0,1], 
                      highlight_color=hcolor,
                      highlight_width=6,
                     zoom=True,
                     zoom_row=[0, 4],
                     zoom_col=[0, 4])
    ax[1].set_xlabel('$i$',fontsize=16)
    ax[1].set_ylabel('$i^\prime$',fontsize=16)
    plt.savefig('./diagrams/dem_two_point_sample5.svg')

    obj = matrix(K, ax=ax[1], type='image', 
                      highlight=True, 
                      highlight_row=[0, 1], 
                      highlight_col=[0,1], 
                      highlight_color=hcolor,
                      highlight_width=7,
                     zoom=True,
                     zoom_row=[0, 2],
                     zoom_col=[0, 2])
    ax[1].set_xlabel('$i$',fontsize=16)
    ax[1].set_ylabel('$i^\prime$',fontsize=16)
    plt.savefig('./diagrams/dem_two_point_sample6.svg')

    obj = matrix(K, ax=ax[1], type='image', 
                      highlight=True, 
                      highlight_row=[0, 1], 
                      highlight_col=[0,1], 
                      highlight_color=hcolor,
                      highlight_width=8,
                     zoom=True,
                     zoom_row=[0, 1],
                     zoom_col=[0, 1])
    ax[1].set_xlabel('$i$',fontsize=16)
    ax[1].set_ylabel('$i^\prime$',fontsize=16)
    plt.savefig('./diagrams/dem_two_point_sample7.svg')

    obj = matrix(K[:2, :2], ax=ax[1], type='values')
    ax[1].set_xlabel('$i$',fontsize=16)
    ax[1].set_ylabel('$i^\prime$',fontsize=16)
    plt.savefig('./diagrams/dem_two_point_sample8.svg')

    ax[0].cla()
    two_point_pred(K, f.T, x, ax=ax[0],ind=[0, 1], stub='dem_two_point_sample', start=9)

    ax[0].cla()
    two_point_pred(K, f.T, x, ax=ax[0],ind=[0, 4], stub='dem_two_point_sample', start=13)


def poisson():
    from scipy.stats import poisson
    fig, ax = plt.subplots(figsize=(14,7))
    y = np.asarray(range(0, 16))
    p1 = poisson.pmf(y, mu=1.)
    p3 = poisson.pmf(y, mu=3.)
    p10 = poisson.pmf(y, mu=10.)

    ax.plot(y, p1, 'r.-', markersize=20, label='$\lambda=1$', lw=3)
    ax.plot(y, p3, 'g.-', markersize=20, label='$\lambda=3$', lw=3)
    ax.plot(y, p10, 'b.-', markersize=20, label='$\lambda=10$', lw=3)
    ax.set_title('Poisson Distribution', fontsize=20)
    ax.set_xlabel('$y_i$', fontsize=20)
    ax.set_ylabel('$p(y_i)$', fontsize=20)
    ax.legend(fontsize=20)
    plt.savefig('./diagrams/poisson.svg')

def logistic():
    fig, ax = plt.subplots(figsize=(14,7))
    f = np.linspace(-8, 8, 100)
    g = 1/(1+np.exp(-f))
    
    ax.plot(f, g, 'r-', lw=3)
    ax.set_title('Logistic Function', fontsize=20)
    ax.set_xlabel('$f_i$', fontsize=20)
    ax.set_ylabel('$g_i$', fontsize=20)
    plt.savefig('./diagrams/logistic.svg')