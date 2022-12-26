import math
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.exceptions import NotFittedError
from matplotlib.pyplot import cm
import torch
import shutil
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.preprocessing import MinMaxScaler


class visualize:
    def __init__(self,path):
        self.idx = 0
        self.data_dict = {}
        self.cont_dict = {}
        self.plot_3d_dict = {}
        self.hist_dict = {}
        self.curv_dict = {}
        self.path = os.path.abspath(path)
        if os.path.exists(self.path):
            print ("Already exists directory %s, will recreate it"%self.path)
            shutil.rmtree(self.path)
        os.makedirs(self.path)
        print("Created directory %s"%self.path)


    def AddPointSet(self,data,title,color):
        self.data_dict[(title,color)] = data

    def AddContour(self,X,Y,Z,title):
        assert X.shape[0] == X.shape[1]
        assert Y.shape[0] == Y.shape[1]
        assert Z.shape[0] == Z.shape[1]
        self.cont_dict[title] = [X,Y,Z]

    def Add3dPlot(self, X, Y, Z, func, title):
        assert X.shape[0] == X.shape[1]
        assert Y.shape[0] == Y.shape[1]
        assert Z.shape[0] == Z.shape[1]
        assert func.shape[0] == func.shape[1]
        self.plot_3d_dict[title] = [X, Y, Z, func]

    def AddCurves(self,x,x_err,dict_val,title):
        # x : float
        # x_err : float or tuple (down, up)
        # dict_val: dict of point y values 
        #   -> key : legend title
        #   -> val : [y,y_err(float or tuple (down, up))]
        # title : title of the plot
        if title not in self.curv_dict.keys():
            self.curv_dict[title]  = dict()
        for key,val in dict_val.items():
            if key not in self.curv_dict[title].keys():
                self.curv_dict[title][key] = (list(),list(),list(),list())
            self.curv_dict[title][key][0].append(x)
            self.curv_dict[title][key][1].append(val[0])
            self.curv_dict[title][key][2].append(x_err)
            self.curv_dict[title][key][3].append(val[1])
        # self.curv_dict = dict of set of curves 
        #     -> key : title for the ax pot
        #     -> val : dict of point sets:
        #                 -> key : legend title
        #                 -> val : tuple of point list ([x],[y],[x_err],[y_err])
        #import pprint
        #pprint.pprint(self.curv_dict)

    def AddHistogram(self,vec,bins,title):
        vec = vec.data.numpy()
        vals, edges = np.histogram(vec,bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        widths = np.diff(edges)
        self.hist_dict[title] = (centers,vals,widths)

    def MakePlot(self,epoch):
        N_data = len(self.data_dict.keys())
        N_cont = len(self.cont_dict.keys())
        N_plot_3d = len(self.plot_3d_dict.keys())
        N_hist = len(self.hist_dict.keys())
        N_curv = len(self.curv_dict.keys())
        Nh = max(N_data, N_cont, N_plot_3d, N_curv, N_hist)
        Nv = int(N_data != 0) + int(N_cont != 0) + int(N_plot_3d != 0) + int(N_hist != 0) + int(N_curv != 0)
        fig, axs = plt.subplots(Nv,Nh,figsize=(Nh*6,Nv*6))
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9, wspace=0.2, hspace=0.2)
        fig.suptitle("Epoch %d"%epoch,fontsize=22)

        if Nv == 1: # Turn the ax vector into array
            axs = axs.reshape(1,-1)
        idx_data = 0
        idx_cont = 0
        idx_3d_plot = 0
        idx_hist = 0
        idx_curv = 0
        idx_vert = 0 
        ##### Data plots #####
        # Print point distribution #
        if len(self.data_dict.keys()) != 0:
            for att,data in self.data_dict.items():
                title = att[0]
                axs[idx_vert,idx_data].scatter(x=data[:,0],y=data[:,1],c=att[1],marker='o',s=1)
                axs[idx_vert,idx_data].set_title(title,fontsize=20)
                axs[idx_vert,idx_data].set_xlim(0,1)
                axs[idx_vert,idx_data].set_ylim(0,1)
                idx_data += 1
            idx_vert += 1

        ##### Contour plots #####
        if len(self.cont_dict.keys()) != 0:
            for title,data in self.cont_dict.items():
                axs[idx_vert,idx_cont].set_title(title,fontsize=20)
                cs = axs[idx_vert,idx_cont].contourf(data[0],data[1],data[2],20)
                #fig.colorbar(cs, ax=axs[idx_vert,idx_cont])
                idx_cont += 1
            idx_vert += 1

        ##### Hist plots ####
        if len(self.hist_dict.keys()) != 0:
            for title,(centers,vals,widths) in self.hist_dict.items():
                axs[idx_vert,idx_hist].set_title(title,fontsize=20)
                axs[idx_vert,idx_hist].bar(centers,vals,align='center',width=widths)
            idx_vert += 1

        ##### 3d plots #####
        if len(self.plot_3d_dict.keys()) != 0:
            for title, data in self.plot_3d_dict.items():
                axs[idx_vert, idx_3d_plot].remove()
                ax = fig.add_subplot(Nv, Nh, idx_vert * Nv + idx_3d_plot, projection='3d')
                ax.set_title(title, fontsize=20)
                cs = ax.scatter(data[0], data[1], data[2], c=data[3], s=20)
                idx_3d_plot += 1
            idx_vert += 1

        ##### Curve plots #####
        if len(self.curv_dict.keys()) != 0:
            for title, set_curves in self.curv_dict.items(): # loop over the different plots
                axs[idx_vert,idx_curv].set_title(title,fontsize=20)
                color = iter(cm.rainbow(np.linspace(0,1,len(set_curves.keys()))))
                for label,curves in set_curves.items(): # Loop over different curves in the same plot
                    # curves = ([x],[y],[x_err],[y_err])
                    c = next(color)
                    axs[idx_vert,idx_curv].errorbar(x = curves[0],
                                             y = curves[1],
                                             xerr = curves[2],
                                             yerr = curves[3],
                                             label = label,
                                             color = c)


            #if self.analytic_integral is not None:
            #    axs[0,idx_data].hlines(y = self.analytic_integral, 
            #                    xmin = 0,
            #                    xmax = self.epochs[-1],
            #                    label = "Analytic value : %0.3f"%self.analytic_integral)

                axs[idx_vert,idx_curv].legend(loc="upper left")
                idx_curv += 1
            idx_vert += 1

     
        # Save fig #
        path_fig = os.path.join(self.path,"frame_%04d.png"%self.idx)
        fig.savefig(path_fig)
        plt.close(fig)
        self.idx += 1
        

class FunctionVisualizer:
    def __init__(self, vis_object: visualize, function, input_dimension, max_plot_dimension):

        self.vis_object = vis_object
        self.function = function
        self.input_dimension = input_dimension
        self.n_components = min(max_plot_dimension, input_dimension)

        assert self.n_components in [2, 3], "plot_dimension can be 2 or 3"
        if self.n_components < self.input_dimension:
            self.use_dimension_reduction = True
            self._init_dimension_reduction()
        else:
            self.use_dimension_reduction = False
        self.grids = self.func_out = self.bins = None
        if function is not None:
            self.grids, self.func_out = self.compute_target_function_grid()

    def _init_dimension_reduction(self):
        self.dimension_transform = LocallyLinearEmbedding(n_components=self.n_components)
        self.scaler = MinMaxScaler()

    def compute_target_function_grid(self):
        """
        generate gird for target function plot
        """
        if self.n_components == 2:
            num_grid_samples = 100**self.n_components
            target_shape = [100] * self.n_components
        elif self.n_components == 3:
            num_grid_samples = 10 ** self.n_components
            target_shape = [10] * self.n_components
        num_samples_per_dimension = math.ceil(num_grid_samples ** (1 / self.input_dimension))
        grid = torch.meshgrid(*[torch.linspace(0, 1, num_samples_per_dimension) for dim in range(self.input_dimension)])
        grid = torch.cat([dim_grid.reshape(-1, 1) for dim_grid in grid], axis=1)[:num_grid_samples]
        func_out = self.function(grid).reshape(target_shape)

        if self.use_dimension_reduction:
            grid = self.dimension_transform.fit_transform(grid)
            grid = self.scaler.fit_transform(grid)
        grids = [grid[:num_grid_samples, dim].reshape(target_shape) for dim in range(self.n_components)]
        return grids, func_out

    def add_target_function_plot(self):
        """
        Add target function plot to visualize object
        """
        if self.n_components == 2:
            self.vis_object.AddContour(*self.grids, self.func_out,
                                       "Target function : " + self.function.name)
        elif self.n_components == 3:
            self.vis_object.Add3dPlot(*self.grids, self.func_out,
                                      "Target function : " + self.function.name)

    def add_trained_function_plot(self, x, plot_name) -> np.ndarray:
        """
        Add trained function plot to visualize object
        return input x or transformed input x
        """
        if self.use_dimension_reduction:
            try:
                visualize_x = self.dimension_transform.transform(x)
                visualize_x = self.scaler.transform(visualize_x)
            except sklearn.exceptions.NotFittedError:
                error_msg = 'Dimension reduction has not trained. ' \
                            'Use the same num_coupling_layers and plot_dimension in server mode'
                raise NotFittedError(error_msg)
        else:
            visualize_x = x

        if self.n_components == 2:
            if self.bins is None:
                bins, x_edges, y_edges = np.histogram2d(visualize_x[:, 0], visualize_x[:, 1], bins=20,
                                                        range=[[0, 1], [0, 1]])
            else:
                newbins, x_edges, y_edges = np.histogram2d(visualize_x[:, 0], visualize_x[:, 1], bins=20,
                                                           range=[[0, 1], [0, 1]])
                self.bins += newbins.T
            x_centers = (x_edges[:-1] + x_edges[1:]) / 2
            y_centers = (y_edges[:-1] + y_edges[1:]) / 2
            x_centers, y_centers = np.meshgrid(x_centers, y_centers)
            self.vis_object.AddContour(x_centers, y_centers, bins, plot_name)
            return visualize_x
        elif self.n_components == 3:
            if self.bins is None:
                bins, (x_edges, y_edges, z_edges) = np.histogramdd(visualize_x, bins=10,
                                                                   range=[[0, 1], [0, 1], [0, 1]])
            else:
                newbins, (x_edges, y_edges, z_edges) = np.histogramdd(visualize_x, bins=10,
                                                                      range=[[0, 1],[0, 1],  [0, 1]])
                self.bins += newbins.T
            x_centers = (x_edges[:-1] + x_edges[1:]) / 2
            y_centers = (y_edges[:-1] + y_edges[1:]) / 2
            z_centers = (z_edges[:-1] + z_edges[1:]) / 2
            x_centers, y_centers, z_centers = np.meshgrid(x_centers, y_centers, z_centers)
            self.vis_object.Add3dPlot(x_centers, y_centers, z_centers, bins, plot_name)
            return visualize_x


class VisualizePoint():

    def __init__(self, index, plot_step=10, path=os.path.join('logs', 'point_plot')):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.bins = None
        self.index = index
        self.plot_step = plot_step
        self.points = []
        self.iteration = 0

    def add_point(self, points: np.ndarray):
        self.points.append(points[int(points.shape[0]*self.index)])
        self.iteration += 1
        if self.iteration % self.plot_step == 0:
            self.plot_points()

    def plot_points(self):
        visualize_x = np.array(self.points)
        bins, x_edges, y_edges = np.histogram2d(visualize_x[:, 0], visualize_x[:, 1], bins=20,
                                                range=[[0, 2 * np.pi], [0, np.pi / 2]])
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        x_centers, y_centers = np.meshgrid(x_centers, y_centers)

        fig, ax = plt.subplots()
        ax.set_title("Point distribution", fontsize=20)
        ax.contourf(x_centers, y_centers, bins, 20)

        path_fig = os.path.join(self.path,"frame_%04d.png"%self.iteration)
        fig.savefig(path_fig)
        plt.close(fig)
