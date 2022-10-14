import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.pyplot import cm
import torch
import shutil

class visualize:
    def __init__(self,path):
        self.idx = 0
        self.data_dict = {}
        self.cont_dict = {}
        self.hist_dict = {}
        self.curv_dict = {}
        self.path = os.path.abspath(path)
        if os.path.exists(self.path):
            print ("Already exists directory %s, will recreate it"%self.path)
            shutil.rmtree(self.path)
        os.makedirs(self.path)
        print ("Created directory %s"%self.path)


    def AddPointSet(self,data,title,color):
        self.data_dict[(title,color)] = data

    def AddContour(self,X,Y,Z,title):
        assert X.shape[0] == X.shape[1]
        assert Y.shape[0] == Y.shape[1]
        assert Z.shape[0] == Z.shape[1]
        self.cont_dict[title] = [X,Y,Z]

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
        N_hist = len(self.hist_dict.keys())
        N_curv = len(self.curv_dict.keys())
        Nh = max(N_data,N_cont,N_curv,N_hist)
        Nv = int(N_data!=0)+int(N_cont!=0)+int(N_hist!=0)+int(N_curv!=0)
        fig, axs = plt.subplots(Nv,Nh,figsize=(Nh*6,Nv*6))
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9, wspace=0.2, hspace=0.2)
        fig.suptitle("Epoch %d"%epoch,fontsize=22)

        if Nv == 1: # Turn the ax vector into array
            axs = axs.reshape(1,-1)
        idx_data = 0
        idx_cont = 0
        idx_hist = 0
        idx_curv = 0
        idx_vert = 0 
        ##### Data plots #####
        # Print point distribution #
        if len(self.data_dict.keys()) != 0:
            for att,data in self.data_dict.items():
                title = att[0]
                color = att[1]
                if len(data) == 2:
                    assert len(color) == 2
                    d1 = data[0]
                    d2 = data[1]
                    axs[idx_vert,idx_data].scatter(x=d1[:,0],y=d1[:,1],c=color[0],marker='o',s=1)
                    axs[idx_vert,idx_data].scatter(x=d2[:,0],y=d2[:,1],c=color[1],marker='o',s=1)
                    axs[idx_vert,idx_data].quiver(d1[:,0],d1[:,1],(d2-d1)[:,0],(d2-d1)[:,1],scale=1,width=0.0005,angles='xy', scale_units='xy')
                else:
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
        
    
#import numpy as np
#a = visualize("animation")
#a.AddPointSet(np.random.rand(100,2),title="TitleA",color="b")
#a.AddPointSet(np.random.rand(100,2),title="TitleB",color="r")
#a.MakePlot()

