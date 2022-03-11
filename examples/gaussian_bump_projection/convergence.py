import numpy as np
import matplotlib.pyplot as plt
import code

'''
The following convergence data is associated
to the first attempt at projection using the DG-MPI
solver. This will include both 2D and 3D meshes 
across various orders.

We choose a Guassian bump as our first function
to test convergence on.
'''
def print_errors(N, errors):
    for i in range(errors.shape[0]-1):
        err = np.log(errors[i+1]/errors[i]) / np.log(N[i+1]/N[i])
        print(err)

Nelem = (np.array([2., 4., 8., 16., 32., 64., 128., 256.]))


'''
CASE: 2D Quadrilaterals
Basis: LagrangeQuad

Arch: CPUs
'''
err_p1 = np.array([
    1.337795915086215439715e-02,
    1.895691624303183281039e-04,
    1.246923726050098665298e-04,
    9.745984408435311968752e-06,
    6.203802403697019534476e-07,
    3.895166755162956505548e-08,
    2.437266873656457305678e-09,
    1.523727676320468068212e-10,
    ])


print('errors p=1')
print_errors(Nelem, err_p1)
# print('errors p=2')
# print_errors(Nelem, err_p2)
# print('errors p=3')
# print_errors(Nelem, err_p3)
# print('errors p=4')
# print_errors(Nelem, err_p4)

Nelem2 = np.array([Nelem[0], Nelem[-1]])
fac = Nelem2[0]/Nelem2[1]
m2_slope = np.array([err_p1[0], err_p1[0]*fac**2])
# m3_slope = np.array([err_p2[0]*3, err_p2[0]*3*fac**3])
# m4_slope = np.array([err_p3[0]*3, err_p3[0]*3*fac**4])
# m5_slope = np.array([err_p4[0]*3, err_p4[0]*3*fac**5])

fig, ax = plt.subplots()
ax.plot(Nelem, err_p1, marker='o', label='$p=1$')
ax.plot(Nelem2, m2_slope, ls='--', color='k')
# ax.plot(Nelem, err_p2, marker='o', label='$p=2$')
# ax.plot(Nelem2, m3_slope, ls='--', color='k')
# ax.plot(Nelem, err_p3, marker='o', label='$p=3$')
# ax.plot(Nelem2, m4_slope, ls='--', color='k')
# ax.plot(Nelem2, m4_slope, ls='--', color='k')
# ax.plot(Nelem, err_p4, marker='o', label='$p=4$')
# ax.plot(Nelem2, m5_slope, ls='--', color='k')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\sqrt{N_\mathrm{elem}}$')
ax.set_ylabel(r'$||\varepsilon_\rho||_2$')
ax.legend()
fig.savefig('conv.png', format='png')
plt.show()

