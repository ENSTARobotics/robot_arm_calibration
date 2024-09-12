#!/usr/bin/python3

#System packages
import os
import sys
import yaml
import glob
import subprocess
from copy import deepcopy

#Utility packages
import numpy as np
from scipy.spatial.transform import Rotation as R

#Custom packages
from robot_arm_calibration.NumericGeometricModel import T, piN, JOINTS_NUMBER, toolName, robotName

### Loading calibration poses
print("Loading configurations...")
try:
    fileName = sys.argv[3]
except:
    try:
        fileName = glob.glob("*Configurations.yaml")[0]
    except:
        print("Invalid configurations file name ! Are you in a proper calibrated model folder or have you specified a correct configurations file name ? ")
        sys.exit(-1)

with open(fileName,"r") as file:
    Configurations = yaml.load(file, Loader=yaml.FullLoader)
    Configurations = np.array(Configurations[list(Configurations.keys())[0]]).reshape((-1,JOINTS_NUMBER))
print("Done !")

print('Number of configurations : ' + str(len(Configurations)))

#plotMode = input("Plot mode ? [mesh, matplotlib, plotly]")
plotMode = "plotly"

if(plotMode == "axes"):

    import matplotlib.pyplot as plt

    ### Tool function
    def plotFrame(transformationMatrix, ax = None):
        """! Plot the frame assiated to an homogeneous 4x4 transformation matrix.

        @param transformationMatrix The tranformation matrix.
        @param ax                   Previously defined matplotlib.pyplot axes - default : None.

        @return  The generated or completed matplotlib.pyplot axes.
        """
        if(ax is None):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        ax.quiver(*transformationMatrix[:3,-1],*np.dot(transformationMatrix,np.array([1,0,0,0]))[:3],color="r",length=0.1,normalize=True)
        ax.quiver(*transformationMatrix[:3,-1],*np.dot(transformationMatrix,np.array([0,1,0,0]))[:3],color="g",length=0.1,normalize=True)
        ax.quiver(*transformationMatrix[:3,-1],*np.dot(transformationMatrix,np.array([0,0,1,0]))[:3],color="b",length=0.1,normalize=True)
        ax.scatter(*transformationMatrix[:3,-1],s=5,color="k")
        return(ax)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    Poses = np.empty((0,3))
    for q in Configurations:
        transformationMatrix = T(q,piN)
        Poses = np.vstack((Poses,transformationMatrix[:3,-1]))
        plotFrame(transformationMatrix,ax)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
    plt.legend()
    plt.show()

elif(plotMode in ["mesh","plotly"]):

    import trimesh

    if(plotMode == "plotly"):
        import plotly.graph_objects as go
        fig = go.Figure()

        def plotMesh(mesh,fig,**kwargs):
            I,J,K = mesh.faces.T
            X,Y,Z = mesh.vertices.T 

            fig.add_trace(go.Mesh3d(
                x=X,
                y=Y,
                z=Z,
                i=I,
                j=J,
                k=K,
                facecolor=mesh.visual.face_colors,
                **kwargs
            ))
    else:
        scene = trimesh.Scene()

    #Tool functions
    def parsePose(pose):
        transform = np.eye(4)
        transform[:3,-1] = np.array([pose["x"],pose["y"],pose["z"]])
        transform[:3,:3] = R.from_euler('xyz', [pose["rx"],pose["ry"],pose["rz"]]).as_matrix()
        return(transform)

    def parseBox(box):
        box = trimesh.creation.box(extents=[box["size"]["dx"],box["size"]["dy"],box["size"]["dz"]],transform=parsePose(box["pose"]))
        box.visual.face_colors = np.array([31,119,180,150])
        return(box)

    def parseCylinder(cylinder):
        cylinder = trimesh.creation.cylinder(radius=cylinder["size"]["radius"],height=cylinder["size"]["height"],transform=parsePose(cylinder["pose"]))
        cylinder.visual.face_colors = np.array([31,119,180,150])
        return(cylinder)
    
    def parseSphere(sphere):
        sphere = trimesh.creation.icosphere(subdivisions=3,radius=sphere["size"]["radius"])
        sphere.apply_transform(parsePose(sphere["pose"]))
        sphere.visual.face_colors = np.array([31,119,180,150])
        return(sphere)

    def parseBorder(border):
        transform = np.eye(4)
        extents = np.zeros(3)
        if(border["axis"] == "x"):
            transform[0,3] = border["offset"]
            extents[0] = 0.001
            extents[1] = 2.0
            extents[2] = 2.0
        elif(border["axis"] == "y"):
            transform[1,3] = border["offset"]
            extents[0] = 2.0
            extents[1] = 0.001
            extents[2] = 2.0
        elif(border["axis"] == "z"):
            transform[2,3] = border["offset"]
            extents[0] = 2.0
            extents[1] = 2.0
            extents[2] = 0.001
        else:
            pass
        border = trimesh.creation.box(extents=extents,transform=transform)
        border.visual.face_colors = np.array([31,119,180,150])
        return(border)

    from urchin import URDF
    import rospkg
    rospack = rospkg.RosPack()

    pkgPath = rospack.get_path("robot_arm_tools")

    command = "bash " + pkgPath + "/scripts/GetRobotDescription.sh 'robot_arm_tools Setup_" + robotName + ".launch' '" + pkgPath + "/config/tools/" + toolName + "/" + toolName + ".urdf.xacro robot_name:=" + robotName + "' ''"
    output = subprocess.run(command,capture_output=True, text=True, shell=True)
    
    if(output.returncode != 0):
        print("Error while getting robot description !")
        sys.exit(-1)

    robotDescription = output.stdout

    import re

    packages = np.unique(np.array([robotDescription[int(m.start()):int(m.end())] for m in re.finditer('package://[a-zA-Z_]+/', robotDescription)]))

    for package in packages:
        robotDescription = robotDescription.replace(package,rospack.get_path(package.split("/")[-2]) + "/")

    robotDescription = robotDescription.replace("file://","")

    with open("/tmp/tmp.urdf","w") as file:
        file.write(robotDescription)
    
    cfg = {}
    Meshes = []

    robot = URDF.load('/tmp/tmp.urdf')

    for k,configuration in enumerate(Configurations):

        print("Configuration " + str(k+1) + "/" + str(len(Configurations)))

        for i,joint in enumerate(robot.actuated_joints):
            cfg[joint.name] = configuration[i]

        axis = trimesh.creation.axis(origin_size=0.01,axis_radius=0.01,axis_length=0.15,transform = robot.link_fk(cfg=cfg)[robot.links[-1]])

        if(plotMode == "mesh"):
            scene.add_geometry(axis)
        # else:
        #     plotMesh(axis,fig)
        
        mesh = trimesh.Trimesh()

        for link in robot.links:
            print("Handling link : " + link.name)

            localMesh = trimesh.Trimesh()
            meshes = deepcopy(robot.visual_trimesh_fk(cfg=cfg,links=[link]))

            if(len(meshes) != 0):
                for submesh in meshes:
                    pose = meshes[submesh]
                    submesh.apply_transform(pose)
                    localMesh = trimesh.util.concatenate([localMesh,submesh])

                Meshes.append(localMesh)
                mesh = trimesh.util.concatenate([mesh,localMesh])

        # newColors = np.array(mesh.visual.face_colors)
        # newColors[:,-1] = 255
        # mesh.visual = trimesh.visual.create_visual(face_colors=newColors,mesh=mesh)
       
        if(plotMode == "mesh"):
            scene.add_geometry(mesh)
        else:
            plotMesh(mesh,fig)

    #Load environment file if provided
    environmentFileName = ""
    try:
        environmentFileName = sys.argv[4]
    except:
        pass

    if(environmentFileName != ""):

        import yaml
        environment = {}
        with open(environmentFileName,"r") as file:
            environment = yaml.load(file, Loader=yaml.FullLoader)

        for key,value in environment.items():
            print("Handling object : " + key)

            if(value["type"] == "box"):
                if(plotMode == "mesh"):
                    scene.add_geometry(parseBox(value))
                else:
                    plotMesh(parseBox(value),fig)
            elif(value["type"] == "cylinder"):
                if(plotMode == "mesh"):
                    scene.add_geometry(parseCylinder(value))
                else:
                    plotMesh(parseCylinder(value),fig)
            elif(value["type"] == "sphere"):
                if(plotMode == "mesh"):
                    scene.add_geometry(parseSphere(value))
                else: 
                    plotMesh(parseSphere(value),fig)
            elif(value["type"] == "border"):
                if(plotMode == "mesh"):    
                    scene.add_geometry(parseBorder(value))
                else:
                    plotMesh(parseBorder(value),fig)
            else:
                continue

    #Add visibility object if needed
    visibilityObjectType = ""
    try:
        visibilityObjectType = sys.argv[5]
    except:
        pass

    if(visibilityObjectType in ["cone","cylinder"]):

        visibilityObject = trimesh.Trimesh()
        if(visibilityObjectType == "cone"):
            visibilityObject = trimesh.creation.cone(radius=0.2,height=0.25,transform=np.dot(robot.link_fk(cfg=cfg)[robot.links[-1]],np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0.25],[0,0,0,1]])))

        elif(visibilityObjectType == "cylinder"):
            visibilityObject = trimesh.creation.cylinder(radius=0.15,height=0.2,transform=np.dot(robot.link_fk(cfg=cfg)[robot.links[-1]],np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.2/2],[0,0,0,1]])))

        visibilityObject.visual.face_colors = np.array([152,223,138,100])
        if(plotMode == "mesh"):
            scene.add_geometry(visibilityObject)
        else:
            plotMesh(visibilityObject,fig)

        for i,mesh in enumerate(Meshes):
            print("Handling mesh : " + str(i+1) + "/" + str(len(Meshes)))

            closest, distance, _ = trimesh.proximity.closest_point(visibilityObject, mesh.vertices)
            if(i == len(Meshes)-1):
                pass
            else:
                if(min(distance) < 0.001):
                    newColors = np.array(mesh.visual.face_colors)
                    newColors = np.array([214,39,40,255])
                    mesh.visual = trimesh.visual.create_visual(face_colors=newColors,mesh=mesh)
                    if(plotMode == "mesh"):
                        scene.add_geometry(mesh)
                    else:
                        plotMesh(mesh,fig)
                else:
                    meshPoint = mesh.vertices[np.argmin(distance)]
                    visibilityPoint = closest[np.argmin(distance)] 
                    segment = trimesh.load_path(np.array([[meshPoint,visibilityPoint]]))
                    segment.colors = np.array([[44,160,44,255]])
                    if(plotMode == "mesh"):
                        scene.add_geometry(segment)
                    else:
                        fig.add_trace(go.Scatter3d(
                            x=[meshPoint[0],visibilityPoint[0]],
                            y=[meshPoint[1],visibilityPoint[1]],
                            z=[meshPoint[2],visibilityPoint[2]],
                            mode='lines',
                            line=dict(color="rgb(44,160,44)", width=8),
                            showlegend=False
                        ))
                        midPoint = (meshPoint + visibilityPoint)/2
                        # fig.add_trace(go.Scatter3d(
                        #     x=[midPoint[0]],
                        #     y=[midPoint[1]],
                        #     z=[midPoint[2]],
                        #     mode='text',
                        #     text=str(np.round(min(distance),3)),
                        #     textfont=dict(color=np.array([[44,160,44,255]]),family="Computer Modern",size=16),
                        #     showlegend=False
                        # ))

    outputFileName = os.path.basename(fileName).split(".")[0]

    if(plotMode == "mesh"):
        if(visibilityObjectType in ["cone","cylinder"]):
            scene.set_camera((0,0.0,0),2.0,(0,0,0))
        else:
            scene.set_camera((np.pi/2,0.0,3*np.pi/4),3.5,(0,0,0.1))
        scene.show()
        data = scene.save_image()
        from PIL import Image
        rendered = Image.open(trimesh.util.wrap_as_stream(data))
        rendered.save(outputFileName + ".png")

        renderedMesh = scene.to_mesh()
        renderedMesh.export(outputFileName + ".obj")
    else:
        #fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False)
        bg = dict(showbackground = False, gridcolor="rgba(0,0,0,0.2)", showticklabels=False, title_text = "")
        fig.update_scenes(xaxis=bg, yaxis=bg, zaxis=bg)

        fig.update_layout(scene_camera=dict(eye=dict(x=1/0.8,y=-1/0.8,z=0.75/0.8)))
        fig.write_html(outputFileName + ".html")
        fig.show()

else:
    print("Invalid plot mode !")
    sys.exit(-1)