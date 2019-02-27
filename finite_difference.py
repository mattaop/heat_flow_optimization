import numpy as np
import matplotlib.pyplot as plt

# room size, m
w = h = 10.
# intervals in x-, y- directions, m
dx = dy = 1
# Thermal diffusivity of steel, m^2 .s-1
D = 19.*10**(-6)

Tcool, Thot = 300, 320

nx, ny = int(w/dx), int(h/dy)

dx2, dy2 = dx*dx, dy*dy
dt = dx2 * dy2 / (2 * D * (dx2 + dy2))

u0 = Tcool * np.ones((nx, ny))
u = u0.copy()

# Initial conditions - ring of inner radius r, width dr centred at (cx,cy) (mm)
r, cx, cy = 2, 5, 5
r2 = r**2
u0[5, 5] = Thot
"""
for i in range(nx):
    for j in range(ny):
        p2 = (i*dx-cx)**2 + (j*dy-cy)**2
        if p2 < r2:
            u0[i,j] = Thot
"""

def do_timestep(u0, u):
    # Propagate with forward-difference in time, central-difference in space
    #print(u0)
    u[1:-1, 1:-1] = u0[1:-1, 1:-1] + D * dt * (
          (u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])/dx2
          + (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])/dy2)
    u[5, 5] = Thot
    u[0, :] = (9*u[1, :]+Tcool)/10
    u[-1, :] = (9*u[-2, :]+Tcool)/10
    u[:, 0] = (9 * u[:, 1] + Tcool) / 10
    u[:, -1] = (9 * u[:, -2] + Tcool) / 10
    u0 = u.copy()
    return u0, u

# Number of timesteps
nsteps = 1001
# Output 4 figures at these timesteps
mfig = [0, 10, 50, 1000]
fignum = 0
fig = plt.figure()
for m in range(nsteps):
    u0, u = do_timestep(u0, u)
    if m in mfig:
        fignum += 1
        print(m, fignum)
        ax = fig.add_subplot(220 + fignum)
        im = ax.imshow(u.copy(), cmap=plt.get_cmap('hot'), vmin=Tcool,vmax=Thot)
        ax.set_axis_off()
        ax.set_title('{:.1f} s'.format(m*dt*1000))
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
cbar_ax.set_xlabel('$T$ / K', labelpad=20)
fig.colorbar(im, cax=cbar_ax)
plt.show()