import numpy as np
import matplotlib.pyplot as plt
import code

'''
The following convergence data represents the
DG-MPI frameworks first convergence study.
We do this for the isentropic vortex case
with 2D and 3D meshes as well as curved meshes.
'''
def print_errors(N, errors):
    for i in range(errors.shape[0]-1):
        err = np.log(errors[i+1]/errors[i]) / np.log(N[i+1]/N[i])
        print(err)

Nelem = (np.array([2., 4., 8., 16., 32., 64., 128., 256.]))

'''
CASE: 2D Quadrilaterals
Basis: LagrangeQuad

Arch: GPUs

NOTE: This is using LaxFriedrichs flux which
doesn't have proper convergence. It does
match the solution from quail though...
'''
# err_p1 = np.array([
#     4.430224143303559264373e-02,
#     4.700260413760024014174e-02,
#     1.634987155936612826523e-02,
#     3.670902501686332860403e-03,
#     7.499912773646258239146e-04,
#     1.960661478671411146399e-04,
#     6.374225326934858219547e-05,
#     9.621563364261404537048e-05
#     ])

# # This one uses hllc
# err_p1_hllc = np.array([
#     4.056602209738374709147e-02,
#     3.657600767002106012526e-02,
#     1.396127186576156739561e-02,
#     3.725234726837679029993e-03,
#     9.052548093709499033022e-04,
#     2.355919624146367015276e-04,
#     1.086704637608227279723e-04,

# ])

# This one uses hllc
err_p1 = np.array([
    4.057487237640151206408e-02,
    3.659091741930729779764e-02,
    1.399040502291195101070e-02,
    3.736915003326653998000e-03,
    9.128245919612695130629e-04,
    2.230394344260663451376e-04,
    5.552798642178519519276e-05,
    1.390242977018099767455e-05


])

err_p2 = np.array([
    3.932754850914681188367e-02,
    1.414430149472130186783e-02,
    3.248199949411992391163e-03,
    3.915751398100587301818e-04,
    4.874570751994351309371e-05,
    5.808150680654376531750e-06,
    7.102447210422075577049e-07,
    7.102447210422075577049e-07, #HOLD FOR NOW

])

err_p3 = np.array([
    2.911726700453628677301e-02,
    6.747203591607888774340e-03,
    5.564165617240108456867e-04,
    3.771907882313834693960e-05, 
    2.369398862192326906355e-06,
    1.424882779046236941575e-07,
    9.074300168426849006228e-09,
    9.074300168426849006228e-09, #HOLD FOR NOW
])

print('errors p=1')
print_errors(Nelem, err_p1)
print('errors p=2')
print_errors(Nelem, err_p2)
print('errors p=3')
print_errors(Nelem, err_p3)
# print('errors p=4')
# print_errors(Nelem, err_p4)

Nelem2 = np.array([Nelem[0], Nelem[-1]])
fac = Nelem2[0]/Nelem2[1]
m2_slope = np.array([err_p1[0], err_p1[0]*fac**2])
m3_slope = np.array([err_p2[0]*3, err_p2[0]*3*fac**3])
m4_slope = np.array([err_p3[0]*3, err_p3[0]*3*fac**4])
# m5_slope = np.array([err_p4[0]*3, err_p4[0]*3*fac**5])

fig, ax = plt.subplots()
ax.plot(Nelem, err_p1, marker='o', label='$p=1$')
ax.plot(Nelem2, m2_slope, ls='--', color='k')
ax.plot(Nelem, err_p2, marker='o', label='$p=2$')
ax.plot(Nelem2, m3_slope, ls='--', color='k')
ax.plot(Nelem, err_p3, marker='o', label='$p=3$')
ax.plot(Nelem2, m4_slope, ls='--', color='k')
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

