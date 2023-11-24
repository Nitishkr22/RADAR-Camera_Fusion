import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize



def create_algebraic_matrix(world_points, projections):
    """
    Create the algebraic matrix A for camera calibration

    Parameters
    -----------
    world points - np.ndarray, shape - (3, n_points)
        points in the world coordinate system

    projections - np.ndarray, shape - (2, n_points)
        projections of the above points in the image

    Returns
    ----------
    A - np.ndarray, shape - (2 * n_points, 12)
        the algebraic matrix used for camera calibration
    """

    assert world_points.shape[1] == projections.shape[1]
    n_points = world_points.shape[1]
    A = np.ones(shape=(2 * n_points, 12))


    c = 0

    ### implement A matrix ######

    for i in range(n_points):

        w = world_points[:, i]
        p = projections[:, i]

        X, Y, Z = w
        u, v = p
        rows = np.zeros(shape=(2, 12))

        ## 1st row of A
        rows[0, 0], rows[0, 1], rows[0, 2], rows[0, 3] = X, Y, Z, 1
        rows[0, 8], rows[0, 9], rows[0, 10], rows[0, 11] = -u * X, -u * Y, -u * Z, -u

        ## 2nd row of A
        rows[1, 4], rows[1, 5], rows[1, 6], rows[1, 7] = X, Y, Z, 1
        rows[1, 8], rows[1, 9], rows[1, 10], rows[1, 11] = -v * X, -v * Y, -v * Z, -v

        A[c : c + 2, :] = rows
        c += 2
        # print("AAAAAAAAAAAA",A)

    return A

def compute_world2img_projection(world_points, M, is_homogeneous=False):
    """
    Given a set of points in the world and the overall camera matrix,
    compute the projection of world points onto the image

    Parameters
    -----------
    world_points - np.ndarray, shape - (3, n_points)
                   points in the world coordinate system

    M - np.ndarray, shape - (3, 4)
        The overall camera matrix which is a composition of the extrinsic and intrinsic matrix

    is_homogeneous - boolean
        whether the coordinates are represented in their homogeneous form
        if False, an extra dimension will  be added for computation

    Returns
    ----------
    projections - np.ndarray, shape - (2, n_points)
                  projections of the world points onto the image
    """


    if not is_homogeneous:
        # convert to homogeneous coordinates
        points_h = np.vstack((world_points, np.ones(world_points.shape[1])))

    h_points_i = M @ points_h  # matrix multiplication

    h_points_i[0, :] = h_points_i[0, :] / h_points_i[2, :]
    h_points_i[1, :] = h_points_i[1, :] / h_points_i[2, :]

    points_i = h_points_i[:2, :]

    return points_i

def geometric_error(m, world_points, projections):
    """
    compute the geometric error wrt the
    prediction projections and the groundtruth projections

    Parameters
    ------------
    m - np.ndarray, shape - (12)
        an 12-dim vector which is to be updated
    world_points - np.ndarray, shape - (3, n)
                   points in the world coordinate system
    projections - np.ndarray(2, n)
                  projections of the points in the image

    Returns
    --------
    error - float
            the geometric error
    """
    
    assert world_points.shape[1] == projections.shape[1]
    error = 0
    n_points = world_points.shape[1]
    for i in range(n_points):
        X, Y, Z = world_points[:, i]
        u, v = projections[:, i]
        u_ = m[0] * X + m[1] * Y + m[2] * Z + m[3]
        v_ = m[4] * X + m[5] * Y + m[6] * Z + m[7]
        d = m[8] * X + m[9] * Y + m[10] * Z + m[11]
        u_ = u_ / d
        v_ = v_ / d
        error += np.sqrt(np.square(u - u_) + np.square(v - v_))
    return error



##### baselor parameters tunnel image 1920x1200 #############
# wrld_pts = np.array([[4.8924,4.0501,4.3433,9.8879,9.2954,9.5949,14.9921,14.8301,14.7197,19.9586,19.5704],
#                     [-0.1093,2.8354,-2.4932,0.03411,3.5679,-2.8090,0.0172,3.1752,-2.6969,0.1358,3.7536],
#                     [0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78]])

# proj_uv = np.array([[389,421,389,336,350,332,315,323,309,302,308],
#                     [600,238,891,577,351,770,583,423,710,578,455]
#                     ])
# n_points = 11
######################################
################################ webcam parameters 640x480 ####################3333

# # wrld_pts = np.array([[10.8980223805051,10.2233201314963,9.34205220924219,9.98352020978178,8.25681320162011,14.1387041197726,5.60228019308513,7.93213103160169,12.0659557896965,13.5663788529394],
# wrld_pts = np.array([[19.0806371652934,19.3161508490297,18.1395883022801,20.2643848656496,19.5918703038088,21.1204413235134,24.3867440956492,23.5924854156177,22.8770447367496,24.9163957186048],
#                     # [0.660116450000464,0.592690454856317,3.1121490117049,-3.47217975305597,5.57237583081003,0.840086004046108,12.6038443346943,-11.6843749683789,6.97065352343798,-4.21080312971452],
#                     [-9.64144157900997,5.07943188721839,8.61802518144892,-2.33589499903679,-5.42804175241498,-13.4943418722534,5.80400717048194,-11.4648213647102,10.2936368818531,-7.56115966573122],
#                     [0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78]])
#                     # ,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78]])

# # proj_uv = np.array([[332,463,173,576,14,330,233,442,49,547],
# proj_uv = np.array([[337,200,81,440,541,330,213,465,91,506],
#                     # [234,238,235,241,231,219,221,223,224,230],
#                     [208,211,213,214,208,206,204,208,206,212]
#                     ])

# # wrld_pts = np.array([[10.8980223805051,10.2233201314963,9.34205220924219,9.98352020978178,8.25681320162011,14.1387041197726,5.60228019308513,7.93213103160169,12.0659557896965,13.5663788529394],
# wrld_pts = np.array([[19.0806371652934,19.3161508490297,18.1395883022801,20.2643848656496,21.1204413235134,23.5924854156177,22.8770447367496,24.9163957186048],
#                     # [0.660116450000464,0.592690454856317,3.1121490117049,-3.47217975305597,5.57237583081003,0.840086004046108,12.6038443346943,-11.6843749683789,6.97065352343798,-4.21080312971452],
#                     [-9.64144157900997,5.07943188721839,8.61802518144892,-2.33589499903679,-13.4943418722534,-11.4648213647102,10.2936368818531,-7.56115966573122],
#                     [0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78]])
#                     # ,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78]])

# # proj_uv = np.array([[332,463,173,576,14,330,233,442,49,547],
# proj_uv = np.array([[337,200,81,440,330,465,91,506],
#                     # [234,238,235,241,231,219,221,223,224,230],
#                     [208,211,213,214,206,208,206,212]
#                     ])

# wrld_pts = np.array([[24.9149454222396,22.8780315244815,24.377837594542,25.0253813444996,19.5929107605539,20.2691562985636,18.131318853781,19.3122086615384,20.008967530365,12.0632816827689,13.5651671712901,
#                       14.0602514920743,14.138784600382,8.26114219711858,9.34235054070211],
#                     [-7.56593719929048,10.2914435191336,5.80188743706908,1.37861989960801,-5.42428494791682,-2.33447592067693,8.61409640896441,5.076415714737,0.867657444370878,6.97528025995376,-4.21470494740116,
#                      -1.70822324661994,0.838730412391801,5.57414433718142,3.11125333682516],
#                     [0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78,
#                      0.78,0.78,0.78,0.78]])

# proj_uv = np.array([[560,91,213,331,541,441,81,200,338,52,547,
#                      442,328,16,173],
#                     [212,205,204,206,217,214,210,211,210,223,228,
#                      224,221,236,236]
#                     ])

# n_points = 15
######################## basler 1st sept 640x480 ############33
wrld_pts = np.array([[10.0198110995233, 8.86439396401329,7.86873382759902,8.59893923864125,14.8572106636809,15.0084875813154,13.4180006763192,19.731423500957,19.9260327417842,
                      20.0721984849806,18.0582842786561,20.8873678174483,24.6985704667786,24.8331548246376,24.9997530087939,23.9365736231739,29.7855082934591,29.9105594927656,30.0152777944594,28.9070950306785,29.3067694128066],
                    [-0.137383947371803,2.96719494564404,3.98744065242041,-4.52877532834497,1.86578875431384,-0.247524353880728,4.0858963377247,-3.0663213337008,2.68107344186584,-0.232869705950027,-5.67533379934974,5.61911753015775,
                     -4.09571799857702,3.10648974606304,-0.450664969236178,-7.61363250831857,-5.4099927159093,3.71252245690607,-1.08250920810619,-7.91346338174415,6.7585857825696],
                    [0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78]])

proj_uv = np.array([[310,185,116,502,255,314,184,374,250,311,438,188,377,254,315,443,384,253,322,424,207],
                    [252,256,262,259,236,237,237,226,226,227,230,226,222,221,223,223,218,218,217,219,217]
                    ])
n_points = 21




fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)


# plotting ground truths
for i in range(n_points):
    ax.scatter(*proj_uv.reshape(-1, 2)[i], color="orange")
    
ax.set_title("projection of points in the image")
plt.show()

A = create_algebraic_matrix(wrld_pts, proj_uv)

A_ = np.matmul(A.T, A)
# compute its eigenvectors and eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(A_)
# find the eigenvector with the minimum eigenvalue
# (numpy already returns sorted eigenvectors wrt their eigenvalues)
# eigenvector is a (12,12) matrix with last vector belongs to eigen vector of minimum eigen values

m = eigenvectors[:, 11]

# reshape m back to a matrix
M1 = m.reshape(3, 4)
# np.save('saved_array.npy', M1)
# print("old",M1)
# M = np.array([[8.72045486e-12, 4.84060721e-12, -8.96633416e-02, 6.99374065e-02],
#                         [2.04313011e-11, -2.31749500e-11, 7.83282213e-01, -6.10960126e-01],
#                         [3.45730887e-14, 9.61667966e-15, 1.28587163e-02, -1.00297987e-02]])

# matrix_list = [
#     [8.72045486e-12, 4.84060721e-12, -8.96633416e-02, 6.99374065e-02],
#     [2.04313011e-11, -2.31749500e-11, 7.83282213e-01, -6.10960126e-01],
#     [3.45730887e-14, 9.61667966e-15, 1.28587163e-02, -1.00297987e-02]
# ]

# M = np.load('saved_array.npy')
# print(( M1==M))
predictions = compute_world2img_projection(wrld_pts, M1, is_homogeneous=False)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
for i in range(n_points):
    if i == 0:
        o_label = "groundtruth"
        g_label = "predictions"
    else:
        o_label = ""
        g_label = ""

    ax.scatter(*proj_uv.reshape(-1, 2)[i], color="orange", alpha=0.75, label=o_label)
    ax.scatter(*predictions.reshape(-1, 2)[i], color="green", alpha=0.75, label=g_label)

ax.set_title("groundtruth vs predictions - direct linear calibration")
ax.legend()
plt.show()

# optimisation process

result = minimize(geometric_error, m, args=(wrld_pts, proj_uv))
# print("Error is: ", result)
M_ = result.x.reshape(3, 4)  # output of optimisation matrix
###############################################################################


# wrld_pts1 = np.array([[9.97952055722003,10.2233201314964],
#                     [-3.45578026460303,0.592690454856319],
#                     [0.78,0.78]])

# proj_uv1 = np.array([[577,333],
#                     [242,235]
#                     ])
predictions_v2 = compute_world2img_projection(wrld_pts, M_, is_homogeneous=False)
#print(M_)

# plot final output##
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
for i in range(n_points):
    if i == 0:
        o_label = "groundtruth"
        g_label = "predictions"
    else:
        o_label = ""
        g_label = ""
        
    ax.scatter(*proj_uv.reshape(-1, 2)[i], color="orange", alpha=0.5, label=o_label)
    ax.scatter(*predictions_v2.reshape(-1, 2)[i], color="green", alpha=0.5, label=g_label)
    
ax.set_title("groundtruth vs predictions - optimization wrt geometric error")
ax.legend()
plt.show()
# print(predictions_v2)
print("Final DLT Matrix: ",M_)

# np.save('dlt_webcam.npy', M_)

J = predictions_v2-proj_uv
# print(J)
row_averages = np.mean(np.abs(J), axis=1)
print("Error in DLT [u,v]: ",row_averages)

distances = np.linalg.norm(proj_uv - predictions_v2, axis=0)
mean_distance_error = np.mean(distances)

print("Mean distance error:", mean_distance_error) 

