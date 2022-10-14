import os 
import sys
import re
import ast
import numpy as np
import subprocess
import torch

from visualize import visualize
import functions

class CatchCuba:
    def __init__(self,func,ndim,nstart,nincrease,maxeval,epsrel):
        self.func        = func
        self.ndim        = ndim
        self.nstart      = nstart
        self.nincrease   = nincrease
        self.maxeval     = maxeval
        self.epsrel      = epsrel
        self.dict_points = {}
        self.dict_values = {}
        self.bins2D      = None

        self._run()

    def _run(self):
        print ("Starting integration with Cuba")
        process = subprocess.Popen(['python','call_cuba.py','-f',self.func,'--ndim',str(self.ndim),'--nstart',str(self.nstart),'--nincrease',str(self.nincrease),'--maxeval',str(self.maxeval),'--epsrel',str(self.epsrel)], stdout=subprocess.PIPE)
        regex = re.compile('[-+]?[\d]+\.?[\d]*[Ee](?:[-+]?[\d]+)?|[-+]?\d*\.\d+|\d+') # ctach scientific, float or int
        points = []
        while True:
            # Get line output #
            output = process.stdout.readline().decode('utf-8')
            # If process has stopped #
            if output == '' and process.poll() is not None:
                rc = process.poll() # Exit code or None if running
                break
            if output:
                line = output.strip()
                if line.startswith("Iteration"):
                    npoints = int(regex.findall(line)[1])
                    print (line)
                elif line.find("+-")!=-1 and line.find("chisq")!=-1:
                    line = line.replace('[1]','')
                    result = regex.findall(line)
                    self.dict_values[npoints] = [float(r) for r in result]
                            # result = [val, err, chi2, dof]
                    print (line)
                elif line.startswith("Point"):
                    line += ' '
                    point = regex.findall(line)
                    points.append([float(p) for p in point])
                        
                elif line.startswith("Results"):
                    results = ast.literal_eval(line.replace("Results ",""))
                    print (line)
                else:
                    print (line)
        # Parse points per evaluation #
        pointer = 0
        for n in self.dict_values.keys():
            self.dict_points[n] = np.array(points[pointer:n])
            pointer = n

    def getPointSets(self):
        return self.dict_points

    def getIntegralValues(self):
        return self.dict_values

    def plotProgress(self):
        self.visObject   = visualize('Plots/Cuba_'+self.func)

        # Add target #
        grid_x1 , grid_x2 = torch.meshgrid(torch.linspace(0,1,100),torch.linspace(0,1,100))
        grid = torch.cat([grid_x1.reshape(-1,1),grid_x2.reshape(-1,1)],axis=1)
        func_out = getattr(functions,self.func)(grid).reshape(100,100)
        self.visObject.AddContour(grid_x1,grid_x2,func_out,"Target function : "+self.func)

        print ("Starting the plotting")
        # Loop over evolution #
        for n in self.dict_values.keys(): 
            print ("\tNumber of points so far : %d/%d"%(n,list(self.dict_values.keys())[-1]))
            val = self.dict_values[n]
                # val = [integral,error,chi2/dof]
            points = self.dict_points[n]

            # Add cloud point #
            self.visObject.AddPointSet(points,title="Cuba output",color='b')

            # 2D cumulative hist #
            if self.bins2D is None:
                self.bins2D, x_edges, y_edges  = np.histogram2d(points[:,0],points[:,1],bins=20,range=[[0,1],[0,1]])
                self.bins2D = self.bins2D.T
            else:
                newbins, x_edges, y_edges = np.histogram2d(points[:,0],points[:,1],bins=20,range=[[0,1],[0,1]])
                self.bins2D += newbins.T

            x_centers = (x_edges[:-1] + x_edges[1:]) / 2
            y_centers = (y_edges[:-1] + y_edges[1:]) / 2
            x_centers, y_centers = np.meshgrid(x_centers,y_centers)

            self.visObject.AddContour(x_centers,y_centers,self.bins2D,"Cuba cumulative points")

            # Integral #
            self.visObject.AddCurves(x     = n,                                                                                                                                                     
                                     x_err = 0,
                                     title = "Integral value",
                                     dict_val = {r'$I$': [val[0],val[1]]})

            self.visObject.AddCurves(x     = n,                                                                                                                                                     
                                     x_err = 0,
                                     title = "$\chi^2/dof$",
                                     dict_val = {r'$chi^2/dof$': [val[2],0]})

            # Plot #
            self.visObject.MakePlot(n)

if __name__ == '__main__':
    from call_cuba import parserCuba
    args = parserCuba()
    instance = CatchCuba(func      = args.function,
                         ndim      = args.ndim,
                         nstart    = args.nstart,
                         nincrease = args.nincrease,
                         maxeval   = args.maxeval,
                         epsrel    = args.epsrel)
    instance.plotProgress()



