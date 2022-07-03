# filters=torch.tensor([1.,-1.],dtype=float).reshape(1,1,2)
import numpy as np
import torch
import time
import math
import torch.nn.functional as F
import tensorflow as tf
from plot_graphs import generate_data
def run_p(filters,frog,k1,k2):
    #with torch.no_grad():
     #   filters = filters.reshape(1, 1, 2)
    Z = 1.
    nx =20  # number of points in the x direction
    ny = 20  # number of points in the y direction
    xmin, xmax = 0.0, 1.0  # limits in the x direction
    ymin, ymax = 0.0, 1.0
    # dt=0.0002
    T = 0.08
    time_steps = 100
    dt = T / time_steps  # limits in the y direction
    lx = xmax - xmin  # domain length in the x direction
    ly = ymax - ymin  # domain length in the y direction
    dx = lx / (nx - 1)  # grid spacing in the x direction
    dy = ly / (ny - 1)  # grid spacing in the y direction
    # E = torch.zeros((nx, ny),dtype=float)  # E_z at t=0
    # Hx = torch.zeros((nx, ny),dtype=float)  # at t=1/2
    # Hy = torch.zeros((nx, ny),dtype=float)  # at t=1/2
    c = math.pi * (np.sqrt(k1 ** 2 + k2 ** 2))
    P = math.pi
    E_a, Hx_a, Hy_a = generate_data(xmax, nx,ny,k1,k2,dx,dy,dt,time_steps)
    E=E_a[0]
    Hx=Hx_a[0]
    Hy=Hy_a[0]
    # for i in range(0, nx):
    #     for j in range(0, ny):
    #         Hx[i, j] = math.sin(c * dt / 2) * (
    #                     -P * k2 * math.sin(P * k1 * dx * i) * math.cos(P * k2 * dy * (j + 1 / 2)) - P * k1 * math.sin(
    #                 P * k2 * dx * i) * math.cos(P * k1 * dy * (j + 1 / 2)))
    #         Hy[i, j] = math.sin(c * dt / 2) * (
    #                     P * k1 * math.cos(P * k1 * dx * (i+1/2)) * math.sin(P * k2 * dy * (j )) + P * k2 * math.cos(
    #                 P * k2 * dx * (i+1/2)) * math.sin(P * k1 * dy * (j )))

    # for i in range(1, nx - 1):
    #     for j in range(1, ny - 1):
    #         E[i, j] = c * (math.sin(P * k1 * dx * i) * math.sin(P * k2 * dy * j) + math.sin(P * k2 * dx * i) * math.sin(
    #             P * k1 * dy * j))

    #E[:, -1] = 0
    #E[-1, :] = 0



    device="cpu"



    # print(conv.weight.device)
    start_time = time.time()
    bc_filters = torch.tensor( [-1., 1.], dtype=float).reshape(1, 1, 2)

    for n in range(time_steps):
        E0=E.detach().clone()
        E01=E.clone()
        Hx0=Hx.detach().clone()
        Hy0=Hy.detach().clone()
        Hx01 = Hx.clone()
        Hy01 = Hy.clone()
        E[1:nx-1,1:ny-1]=amper(E0,Hx0,Hy0,Z,dt,dx,dy,nx,ny,bc_filters,1)
        #print(E.requires_grad)
        #E[1:nx - 1, 1:ny - 1] = amper(E0, Hx0, Hy0, Z, dt, dx, dy, nx, ny, bc_filters, 1)
        E[frog:nx - frog, frog:ny - frog] = amper(E01,Hx0,Hy0,Z,dt,dx,dy,nx,ny,filters,frog)
        Em=E.detach().clone()
        Em1 = E.clone()
        Hx[1:nx - 1, 0:ny -1] = faraday(Em, Hx0, Hy0, Z, dt, dx, dy, nx, ny, bc_filters, 1)[0]
        Hy[0:nx - 1, 1:ny - 1] = faraday(Em, Hx0, Hy0, Z, dt, dx, dy, nx, ny, bc_filters, 1)[1]
        Hx[frog:nx - frog, frog-1:ny - frog] = faraday(Em,Hx01,Hy0,Z,dt,dx,dy,nx,ny,filters,frog)[0]
        Hy[frog-1:nx - frog, frog:ny - frog] = faraday(Em1,Hx01,Hy0,Z,dt,dx,dy,nx,ny,filters,frog)[1]
        if n % 10 == 0:
            for i in range(frog, nx - frog):
                for j in range(frog, ny - frog):
                    E_a[i, j] = c * math.cos(c * (n+1) * dt) * (
                                math.sin(P * k1 * dx * i) * math.sin(P * k2 * dy * j) + math.sin(
                            P * k2 * dx * i) * math.sin(P * k1 * dy * j))
            # print(E_a.dtype)
            print("{:.6f}".format((torch.square(E[frog:nx - frog, frog:ny - frog]-E_a[frog:nx - frog, frog:ny - frog])).mean()))



    print("--- %s seconds ---" % (time.time() - start_time))
    # print(filters[0,0])
    #with torch.no_grad():
     #   filters = filters.reshape( 2)
      #  print(filters.requires_grad)
    return 1
def amper(E,Hnx,Hny,Z,dt,dx,dy,nx,ny,filters,frog):
    S1 = (Z * dt / dx) * F.conv1d(torch.transpose(Hny, 0, 1).reshape(ny, 1, nx),filters).reshape(ny, nx - (2*frog-1)).transpose(1, 0)[
                         0:-1, frog:ny - frog]
    S2 = (Z * dt / dy) * (F.conv1d(Hnx.reshape(nx, 1, ny),filters).reshape(nx, ny-(2*frog-1)))[frog:nx - frog, 0:-1]
    return E[frog:nx - frog, frog:ny - frog] + S1 - S2


def faraday(E,Hnx,Hny,Z,dt,dx,dy,nx,ny,filters,frog):
    S3 = (dt / (Z * dy)) * F.conv1d(E.reshape(nx, 1, ny),filters).reshape(nx, ny - (2*frog-1))[frog:nx - frog, 0:]
    #print('hhh')
    S4 = (dt / (Z * dx)) * F.conv1d(torch.transpose(E, 0, 1).reshape(ny, 1, nx),filters).reshape(ny, nx -(2*frog-1)).transpose(1, 0)[0:,
                           frog:ny - frog]

    Ax= Hnx[frog:nx - frog, frog-1:ny - frog] - S3
    Ay= Hny[frog-1:nx - frog, frog:ny - frog] + S4
    return [Ax,Ay]

