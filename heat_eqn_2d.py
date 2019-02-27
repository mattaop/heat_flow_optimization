import numpy as np
import matplotlib.pyplot as plt


def initialize_grid(x_len, y_len, Nx, Ny, init_temp, heater_temp, heater_placement):
    dx, dy = x_len/(Nx-1), y_len(Ny-1)
    u0 = np.ones((Nx, Ny))*init_temp
    dx2, dy2 = dx * dx, dy * dy
    dt = min(dx2 * dy2 / (2 * D * (dx2 + dy2)), 10)  # set dt to the minimum of 10 and max_dt to obtain stable solution
    print("dt = {}".format(dt))
    u0[heater_placement] = heater_temp
    u = u0.copy()
    return u0, u, dx, dy, dt


def calculate_temperature_at_new_timestep_cds(u0, u, dt, dx, dy, out_temp, heater_placement, heater_temp):
    # Propagate with forward-difference in time, central-difference in space
    u[1:-1, 1:-1] = u0[1:-1, 1:-1] + D * dt * ((u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])/dx**2 + (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])/dy**2)

    u[heater_placement] = heater_temp
    u[0, :] = (9*u[1, :]+out_temp)/10
    u[-1, :] = (9*u[-2, :]+out_temp)/10
    u[:, 0] = (9*u[:, 1]+out_temp)/10
    u[:, -1] = (9*u[:, -2]+out_temp)/10

    u0 = u.copy()

    return u0, u
# Thermal diffusivity of air, m2.s-1
D = 19*10**(-6)

Tcool, Thot, Tout = 15+273, 25+273, 15+273

placement = ((5, 5))

u0, u, dx, dy, dt = initialize_grid(4, 4, 15, 15, Tcool, Thot, placement)

# Number of timesteps
timeEnd = dt*10001
nsteps = int(timeEnd/dt)
# Output 4 figures at these timesteps
tList = dt*np.array([0, 1000, 5000, 10000])
#mfig = [0, 10, 50, 10000]
fignum = 0
fig = plt.figure()
for m in range(nsteps):
    t = dt*m
    if t in tList:
        fignum += 1
        print(t, fignum)
        ax = fig.add_subplot(220 + fignum)
        im = ax.imshow(u.copy(), cmap=plt.get_cmap('hot'), vmin=Tcool,vmax=Thot)
        ax.set_axis_off()
        #ax.set_title('{:.1f} m'.format(t/60))
    u0, u = calculate_temperature_at_new_timestep_cds(u0, u, dx, dy, Tout, placement, Thot)
print(u0)
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
cbar_ax.set_xlabel('$T$ / K', labelpad=20)
fig.colorbar(im, cax=cbar_ax)
plt.show()