# -*- coding: utf-8 -*-
"""
Created on 07/12/2021

@author: Casey Rodgers

Create Non Structural Damage Material Textures for Blender
    
References: 
// Nonstructural Damage
    -Seismic Performance Assessment of Buildings, Volume 2-Implementation Guide, Second Edition, FEMA P-58-2, December 2018. 2019. 
    -“How to Add a New Stop to the Color Ramp?” Blender Stack Exchange, 1 Sept. 2020 https://blender.stackexchange.com/questions/189712/how-to-add-a-new-stop-to-the-color-ramp. 
    -“Make a Photo-Realistic Concrete Material with Cracks in Blender 2.82.” YouTube, YouTube, 6 Mar. 2020, https://www.youtube.com/watch?v=8Odon-JrQ7o. 
    -“Blender Tutorial - Procedural Cracked Surface Material.” YouTube, YouTube, 13 Oct. 2020, https://www.youtube.com/watch?v=XFDQrdRmDwc. 
    -“Shadernode(NodeInternal).” ShaderNode(NodeInternal) - Blender Python API, 17 Dec. 2021, https://docs.blender.org/api/current/bpy.types.ShaderNode.html. 
    - https://blender.stackexchange.com/questions/23436/control-cycles-eevee-material-nodes-and-material-properties-using-python

"""

## Imports
import bpy
import numpy as np
from scipy import special
#import matplotlib.pyplot as plt


""" Helper Functions """

""" Update Color Ramp """
# Update color ramp by adding colors at specific positions
# colorRamp = color ramp object
# colors = array of colors wanted (one 4-tuple per stop)
# positions = array of positions wanted (one float per stop)
def updateColorRamp(colorRamp, colors, positions):

    # Go through each stop we want
    for i in range(len(colors)):
        
        # Use existing stop or create a new one
        if (i == 0 or i == 1):
            colorRamp.elements[i].position = positions[i]  # Set position of existing stop
        else:
            colorRamp.elements.new(positions[i])    # Create new stop
        
        colorRamp.elements[i].color = colors[i]     # Set color
        


""" Calculate Damage Percent """
# Calculate percent area of damage based on fragility curves
# Fragility curves for precast concrete non-structural panels
# Curves are from FEMA P-58 Seismic Performance Assessment of Buildings Vol 2
# Curves are lognormal per page 201 and parameters are given by page 66
# In-plane damage uses story drift ratio as the damage parameter
# Out of plane damage uses acceleration as the damage parameter
# inOrOut = in-plane damage or out of plane damage? 0 = in. 1 = out
# damagePara = damage parameter (Story drift or acceleration)
# connHeight = connection height (m)
# strHeight = structure height above grade (m)
# Returns damage percent
def calcDamagePercent(inOrOut, damagePara, connHeight, strHeight):

    damagePer = -1;     # Damage percent
    
    # In-Plane damage
    if (inOrOut == 0):
        median_val = 0.019 + 0.005;  # Median Demand (Table 2-8)(Risk category III)
        disp_val = 0.5;                 # Dispersion Value 
        mean_val = np.log(median_val)   # Mean value 
        
        part1 = -(np.log(damagePara) - mean_val) / (disp_val * np.sqrt(2))
        damagePer = 0.5 * special.erfc(part1)
        
        
    # Out of Plane damage
    elif (inOrOut == 1):
        # Find median value
        Sds = 1.2;          # Short period ground motion coefficient
        Ip = 1.0;           # Component importance factor (ASCE 13.1.3)
        ap = 1.0;           # Component application factor (ASCE Table 13.5-1)
        Rp = 2.5;           # Force reduction factor (ASCE Table 13.5-1)
        h = strHeight;           # Structure height above grade (in ft)
        z = connHeight;          # Top of wall height (in ft)
        
        Fp = 0.4 * Sds * Ip * ap / Rp * (1 + 2 * z / h)     # Equivalent static force to determine required anchorage strength
        minFp = 0.3 * Sds * Ip      # Min Fp
        maxFp = 1.65 * Sds * Ip     # Max Fp
        
        if (Fp < minFp):
            Fp = minFp
        elif (Fp > maxFp):
            Fp = maxFp
        
        phi_Rn = Fp         # Factored strength
        median_val = 2.04 * phi_Rn;  # Median Demand (FEMA Page 239)
        
        # Calculate damage percent
        disp_val = 0.5;                 # Dispersion Value 
        mean_val = np.log(median_val)   # Mean value
        part1 = -(np.log(damagePara) - mean_val) / (disp_val * np.sqrt(2))
        damagePer = 0.5 * special.erfc(part1)
        
        
    return damagePer



""" Plot Fragility Curves """
# Plot fragility curves based on how they are now
def plotFragilityCurves():
    
    # Initialize Variables
    x1 = np.linspace(0.001, 0.1, 1000)  # In-Plane damage
    y1 = np.zeros(len(x1))
    
    x2 = np.linspace(0.001, 2.5, 1000)  # Out of Plane damage
    y2 = np.zeros(len(x1))
    
    # Calculate probability of failure
    for i in range(len(x1)):
        y1[i] = calcDamagePercent(0, x1[i])
        y2[i] = calcDamagePercent(1, x2[i])
        
    # Create plots
    # plt.plot(x1, y1)
    # plt.xlabel('Story Drift Ratio')
    # plt.ylabel('Probability of Failure')
    # plt.title('In-Plane Damage Fragility Curve')
    # plt.grid(True)
    # plt.show()
    
    # plt.plot(x2, y2)
    # plt.xlabel('Peak Floor Acceleration (g)')
    # plt.ylabel('Probability of Failure')
    # plt.title('Out of Plane Damage Fragility Curve')
    # plt.grid(True)
    # plt.show()



############################################
############################################
""" Create material based on percent damage given """
# Creates damaged blender material based on percent damage given
# Structure Damage and Material Images must be in the same folder "texFolder"
# inOrOut = in-plane damage or out of plane damage? 0 = in. 1 = out
# damagePara = damage parameter (Story drift or acceleration)
# connHeight = connection height (m)
# strHeight = structure height above grade (m)
# texFolder = folder with the textures
# matImgName = inputted material textures folder

def createDamageMaterial(inOrOut, damagePara, connHeight, strHeight, texFolder,  matImgName):
    
    """ Create Material """
    
    # strImgName = inputted structure damage textures folder    
    #strFolder = texFolder + '\\' + strImgName    # Structure damage image name prefix
    matFolder = texFolder + '\\' + matImgName    # Structure material image name prefix
    
    # Parse together name based on in or out of plane damage and damage parameter
    mat_name = 'Wall Damage '                   # Material Name
    if (inOrOut == 0):
        mat_name = mat_name + 'InPlane '
    elif (inOrOut == 1):
        mat_name = mat_name + 'OutPlane '
        
    mat_name = mat_name + str(damagePara)
    
    if (inOrOut == 1):
        mat_name = mat_name + 'g'
        
        
    # Check if material exists and create new one if not
    if mat_name in bpy.data.materials:
        mat = bpy.data.materials[mat_name]      # Get existing material
    else:
        mat = bpy.data.materials.new(mat_name)     # Create material
    
    mat.use_nodes = True            # Use nodes
    nodes = mat.node_tree.nodes     # Set material nodes to name nodes
    nodes.clear()                   # Clear all nodes to start clean
    links = mat.node_tree.links     # Set material links to name links
    
    space_w = 100              # Amount of width space between nodes
    randScale = 300             # Random scaling variable
    
    
    # Calculate Damage Percent
    percent = calcDamagePercent(inOrOut, damagePara, connHeight, strHeight)       # Damage percent 
    print(percent)
    
    
    
    ############################################
    """ Concrete Color and Roughness """
    
    curr_x0c = 0                  # Current width (for location) for texture coord
    curr_y0c = 0                 # Current height (for location) for texture coord
    
    
    ## Create Texture Coordinate Node 1
    # Properties
    node_texCoor1c = nodes.new(type='ShaderNodeTexCoord')   # Create node
    node_texCoor1c.location = curr_x0c, curr_y0c                # Location
    curr_x0c = curr_x0c + node_texCoor1c.width + space_w       # Update current x value
    
    
    ## Create Mapping Node 1
    # Properties
    node_map1c = nodes.new(type='ShaderNodeMapping')   # Create node
    node_map1c.vector_type = 'TEXTURE'         # Vector Type
    node_map1c.location = curr_x0c, curr_y0c                # Location
    curr_x0c = curr_x0c + node_map1c.width + space_w       # Update current x value
    
    # Inputs
    node_map1c.inputs['Location'].default_value[0] = np.random.rand() * randScale  # Location x
    node_map1c.inputs['Location'].default_value[1] = np.random.rand() * randScale  # Location y
    node_map1c.inputs['Location'].default_value[2] = np.random.rand() * randScale  # Location z
    link1_map1c = links.new(node_texCoor1c.outputs['Generated'], node_map1c.inputs['Vector'])  # Vector
    
    
    ### Dirt
    curr_x1c = curr_x0c
    curr_y1c = curr_y0c
    
    ## Create Noise Texture
    # Properties
    node_noise1c = nodes.new(type='ShaderNodeTexNoise')   # Create node
    node_noise1c.noise_dimensions = '3D'                  # Dimensions
    node_noise1c.location = curr_x1c, curr_y1c                       # Location
    curr_x1c = curr_x1c + node_noise1c.width + space_w       # Update current x value
    
    # Inputs
    #print(node_noise.inputs.keys())
    node_noise1c.inputs['Scale'].default_value = 5.0        # Scale
    node_noise1c.inputs['Detail'].default_value = 2.0       # Detail
    node_noise1c.inputs['Roughness'].default_value = 0.5     # Roughness
    node_noise1c.inputs['Distortion'].default_value = 0.0    # Distortion
    link1_noise1c = links.new(node_map1c.outputs['Vector'], node_noise1c.inputs['Vector'])  # Vector
    
    
    ## Create Color Ramp Node
    # Properties
    node_cr1c = nodes.new(type='ShaderNodeValToRGB')     # Create node
    colorRamp1c = node_cr1c.color_ramp                    # Color ramp
    cr1c_colors = [(0.0, 0.0, 0.0, 1), (0.037, 0.016, 0.02, 1)]   # Color ramp colors
    cr1c_pos = [0.508, 1.0]     # Color ramp positions
    updateColorRamp(colorRamp1c, cr1c_colors, cr1c_pos)    # Add pos and colors to color ramp
    colorRamp1c.color_mode = 'RGB'           # Color mode
    colorRamp1c.hue_interpolation = 'NEAR'   # Color interpolation
    colorRamp1c.interpolation = 'LINEAR'     # Interpolation between color stops
    node_cr1c.location = curr_x1c, curr_y1c         # Location
    curr_x1c = curr_x1c + node_cr1c.width + space_w       # Update current x value
    
    # Inputs
    link1_cr1c = links.new(node_noise1c.outputs['Fac'], node_cr1c.inputs['Fac'])  # Fac
    
    
    ### Concrete
    curr_x2c = curr_x0c
    curr_y2c = curr_y1c - 2 * node_noise1c.height - space_w
    
    ## Create Image Texture Node 1
    # Properties
    node_imgTex1c = nodes.new(type='ShaderNodeTexImage')   # Create node
    node_imgTex1c.extension = 'REPEAT'         # Image extension
    node_imgTex1c.image = bpy.data.images.load(matFolder + '_Color.png')         # Image
    node_imgTex1c.interpolation = 'Linear'         # Image interpolation
    node_imgTex1c.projection = 'BOX'         # Image projection
    node_imgTex1c.location = curr_x2c, curr_y2c                # Location
    curr_x2c = curr_x2c + node_imgTex1c.width + space_w       # Update current x value
    
    # Inputs
    link1_imgTex1c = links.new(node_map1c.outputs['Vector'], node_imgTex1c.inputs['Vector'])  # Color
    
    
    ## Create RGB Curves Node 1
    # Properties
    node_rgbCurve1c = nodes.new(type='ShaderNodeRGBCurve')     # Create node
    curveMap1c = node_rgbCurve1c.mapping             # Curve Mapping
    curveMap1c.tone = 'STANDARD'                    # Tone of the curve
    curve1c = curveMap1c.curves[3]                   # Curve 'C'
    curve1c.points.new(0.35625, 0.65)            # Add curve point
    node_rgbCurve1c.location = curr_x2c, curr_y2c            # Location
    curr_x2c = curr_x2c + node_rgbCurve1c.width + space_w       # Update current x value
    
    # Inputs
    node_rgbCurve1c.inputs['Fac'].default_value = 1.0       # Fac
    link1_rgbCurve1c = links.new(node_imgTex1c.outputs['Color'], node_rgbCurve1c.inputs['Color'])  # Color
    
    
    ## Create RGB Curves Node 2
    # Properties
    node_rgbCurve2c = nodes.new(type='ShaderNodeRGBCurve')     # Create node
    curveMap2c = node_rgbCurve2c.mapping             # Curve Mapping
    curveMap2c.tone = 'STANDARD'                    # Tone of the curve
    curve2c = curveMap2c.curves[3]                   # Curve 'C'
    curve2c.points.new(0.65, 0.35625)            # Add curve point
    node_rgbCurve2c.location = curr_x2c, curr_y2c - 100            # Location
    curr_x2c = curr_x2c + node_rgbCurve2c.width + space_w       # Update current x value
    
    # Inputs
    node_rgbCurve2c.inputs['Fac'].default_value = 1.0       # Fac
    link1_rgbCurve2c = links.new(node_rgbCurve1c.outputs['Color'], node_rgbCurve2c.inputs['Color'])  # Color
    
    
    ### Roughness
    curr_x3c = curr_x0c
    curr_y3c = curr_y2c - 2 * node_imgTex1c.height - space_w
    
    ## Create Image Texture Node 2
    # Properties
    node_imgTex2c = nodes.new(type='ShaderNodeTexImage')   # Create node
    node_imgTex2c.extension = 'REPEAT'         # Image extension
    node_imgTex2c.image = bpy.data.images.load(matFolder + '_Roughness.png')         # Image
    node_imgTex2c.interpolation = 'Linear'         # Image interpolation
    node_imgTex2c.projection = 'BOX'         # Image projection
    node_imgTex2c.location = curr_x3c, curr_y3c                # Location
    curr_x3c = curr_x3c + node_imgTex2c.width + space_w       # Update current x value
    
    # Inputs
    link1_imgTex2c = links.new(node_map1c.outputs['Vector'], node_imgTex2c.inputs['Vector'])  # Color
    
    
    
    
    ############################################
    """ Concrete Surface Normals """
    
    curr_x1n = curr_x0c              # Reset current x
    curr_y1n = curr_y3c - 2 * node_imgTex2c.height - space_w       # Update current y value
    
    ## Create Image Texture Node 1
    # Properties
    node_imgTex1n = nodes.new(type='ShaderNodeTexImage')   # Create node
    node_imgTex1n.extension = 'REPEAT'         # Image extension
    node_imgTex1n.image = bpy.data.images.load(matFolder + '_Normal.png')         # Image
    node_imgTex1n.interpolation = 'Linear'         # Image interpolation
    node_imgTex1n.projection = 'BOX'         # Image projection
    node_imgTex1n.location = curr_x1n, curr_y1n                # Location
    curr_x1n = curr_x1n + node_imgTex1n.width + space_w       # Update current x value
    
    # Inputs
    link1_imgTex1n = links.new(node_map1c.outputs['Vector'], node_imgTex1n.inputs['Vector'])  # Color
    
    
    ## Normal Map Node
    # Properties
    node_normMap1n = nodes.new(type='ShaderNodeNormalMap')   # Create node
    node_normMap1n.space = 'OBJECT'         # Space of the normal
    node_normMap1n.location = curr_x1n, curr_y1n                # Location
    curr_x1n = curr_x1n + node_normMap1n.width + space_w       # Update current x value
    
    # Inputs
    link1_normMap1n = links.new(node_imgTex1n.outputs['Color'], node_normMap1n.inputs['Color'])  # Color
    
    
    
    
    ############################################
    """ Concrete Displacement """
    
    curr_x1d = curr_x0c              # Reset current x
    curr_y1d = curr_y1n - 2 * node_imgTex1n.height - space_w       # Update current y value
    
    ## Create Image Texture Node 1
    # Properties
    node_imgTex1d = nodes.new(type='ShaderNodeTexImage')   # Create node
    node_imgTex1d.extension = 'REPEAT'         # Image extension
    node_imgTex1d.image = bpy.data.images.load(matFolder + '_Displacement.png')    # Image
    node_imgTex1d.interpolation = 'Linear'         # Image interpolation
    node_imgTex1d.projection = 'BOX'         # Image projection
    node_imgTex1d.location = curr_x1d, curr_y1d                # Location
    curr_x1d = curr_x1d + node_imgTex1d.width + space_w       # Update current x value
    
    # Inputs
    link1_imgTex1d = links.new(node_map1c.outputs['Vector'], node_imgTex1d.inputs['Vector'])  # Color
    
    
    ## Math Node - Multiply
    # Properties
    node_math1d = nodes.new(type='ShaderNodeMath')   # Create node
    node_math1d.operation = 'MULTIPLY'         # Operation
    node_math1d.use_clamp = False             # Use clamp
    node_math1d.location = curr_x1d, curr_y1d                # Location
    curr_x1d = curr_x1d + node_math1d.width + space_w       # Update current x value
    
    # Inputs
    link1_math1d = links.new(node_imgTex1d.outputs['Color'], node_math1d.inputs[0])  # Value 1
    node_math1d.inputs[1].default_value = 0.1                                        # Value 2
    
    
    
    
    ############################################
    """ Concrete Spalling """
    
    curr_x1s = 0              # Reset current x
    curr_y1s = curr_y1d - 2 * node_imgTex1d.height - space_w       # Update current y value
    
    
    ## Create Texture Coordinate Node 1
    # Properties
    node_texCoor1s = nodes.new(type='ShaderNodeTexCoord')   # Create node
    node_texCoor1s.location = curr_x1s, curr_y1s               # Location
    curr_x1s = curr_x1s + node_texCoor1s.width + space_w       # Update current x value
    
    
    ## Create Mapping Node 1
    # Properties
    node_map1s = nodes.new(type='ShaderNodeMapping')   # Create node
    node_map1s.vector_type = 'TEXTURE'         # Vector Type
    node_map1s.location = curr_x1s, curr_y1s                # Location
    curr_x1s = curr_x1s + node_map1s.width + space_w       # Update current x value
    
    # Inputs
    node_map1s.inputs['Location'].default_value[0] = np.random.rand() * randScale  # Location x
    node_map1s.inputs['Location'].default_value[1] = np.random.rand() * randScale  # Location y
    node_map1s.inputs['Location'].default_value[2] = np.random.rand() * randScale  # Location z
    link1_map1s = links.new(node_texCoor1s.outputs['Generated'], node_map1s.inputs['Vector'])  # Vector
    
    
    ## Create Noise Texture
    # Properties
    node_noise1s = nodes.new(type='ShaderNodeTexNoise')   # Create node
    node_noise1s.noise_dimensions = '3D'                  # Dimensions
    node_noise1s.location = curr_x1s, curr_y1s                       # Location
    curr_x1s = curr_x1s + node_noise1s.width + space_w       # Update current x value
    
    # Inputs
    #print(node_noise.inputs.keys())
    node_noise1s.inputs['Scale'].default_value = 5.0        # Scale
    node_noise1s.inputs['Detail'].default_value = 16.0       # Detail
    node_noise1s.inputs['Roughness'].default_value = 0.5     # Roughness
    node_noise1s.inputs['Distortion'].default_value = 0.0    # Distortion
    link1_noise1s = links.new(node_map1s.outputs['Vector'], node_noise1s.inputs['Vector'])  # Vector
    
    
    ## Create Voronoi Texture 
    # Properties
    node_vor1s = nodes.new(type='ShaderNodeTexVoronoi')   # Create node
    node_vor1s.distance = 'EUCLIDEAN'         # Distance
    node_vor1s.feature = 'SMOOTH_F1'          # Feature
    node_vor1s.voronoi_dimensions = '3D'      # Dimensions
    node_vor1s.location = curr_x1s, curr_y1s                # Location
    curr_x1s = curr_x1s + node_vor1s.width + space_w       # Update current x value
    
    # Inputs
    #print(node_vor.inputs.keys())
    node_vor1s.inputs['Scale'].default_value = 2.5           # Scale
    node_vor1s.inputs['Smoothness'].default_value = 1.0        # Smoothness
    node_vor1s.inputs['Randomness'].default_value = 0.0        # Randomness
    link1_vor1s = links.new(node_noise1s.outputs['Color'], node_vor1s.inputs['Vector'])  # Vector
    
    
    ## Create Color Ramp Node
    # Properties
    node_cr1s = nodes.new(type='ShaderNodeValToRGB')     # Create node
    colorRamp1s = node_cr1s.color_ramp                    # Color ramp
    cr1s_colors = [(1.0, 1.0, 1.0, 1), (0.0, 0.0, 0.0, 1)]   # Color ramp colors
    cr1s_firstPos = 0.584 * (1 - percent)
    cr1s_pos = [cr1s_firstPos, 0.584]     # Color ramp positions
    updateColorRamp(colorRamp1s, cr1s_colors, cr1s_pos)    # Add pos and colors to color ramp
    colorRamp1s.color_mode = 'RGB'           # Color mode
    colorRamp1s.hue_interpolation = 'NEAR'   # Color interpolation
    colorRamp1s.interpolation = 'LINEAR'     # Interpolation between color stops
    node_cr1s.location = curr_x1s, curr_y1s         # Location
    curr_x1s = curr_x1s + node_cr1s.width + space_w       # Update current x value
    
    # Inputs
    link1_cr1s = links.new(node_vor1s.outputs['Distance'], node_cr1s.inputs['Fac'])  # Fac
    
    
    
    
    ############################################
    """ Concrete Finer Cracks """
    
    curr_x0f = 0              # Reset current x
    curr_y0f = curr_y1s - 2 * node_map1s.height - 2 * space_w       # Update current y value
    
    curr_x1f = 0              # Reset current x
    curr_y1f = curr_y0f       # Update current y value
    
    curr_x2f = 0              # Reset current x
    curr_y2f = curr_y0f       # Update current y value
    
    
    ## Create Texture Coordinate Node 1
    # Properties
    node_texCoor1f = nodes.new(type='ShaderNodeTexCoord')   # Create node
    node_texCoor1f.location = curr_x0f, curr_y0f                # Location
    curr_x0f = curr_x0f + node_texCoor1f.width + space_w       # Update current x value
    
    
    ## Create Mapping Node 1
    # Properties
    node_map1f = nodes.new(type='ShaderNodeMapping')   # Create node
    node_map1f.vector_type = 'TEXTURE'         # Vector Type
    node_map1f.location = curr_x0f, curr_y0f                # Location
    curr_x0f = curr_x0f + node_map1f.width + space_w       # Update current x value
    
    # Inputs
    node_map1f.inputs['Location'].default_value[0] = np.random.rand() * randScale  # Location x
    node_map1f.inputs['Location'].default_value[1] = np.random.rand() * randScale  # Location y
    node_map1f.inputs['Location'].default_value[2] = np.random.rand() * randScale  # Location z
    link1_map1f = links.new(node_texCoor1f.outputs['Generated'], node_map1f.inputs['Vector'])  # Vector
    
    
    ### Cracks
    curr_x1f = curr_x0f
    curr_y1f = curr_y0f
    
    ## Create Noise Texture 1
    # Properties
    node_noise1f = nodes.new(type='ShaderNodeTexNoise')   # Create node
    node_noise1f.noise_dimensions = '3D'                  # Dimensions
    node_noise1f.location = curr_x1f, curr_y1f                       # Location
    curr_x1f = curr_x1f + node_noise1f.width + space_w       # Update current x value
    
    # Inputs
    #print(node_noise.inputs.keys())
    node_noise1f.inputs['Scale'].default_value = 0.5        # Scale
    node_noise1f.inputs['Detail'].default_value = 16.0       # Detail
    node_noise1f.inputs['Roughness'].default_value = 0.45     # Roughness
    node_noise1f.inputs['Distortion'].default_value = 0.0    # Distortion
    link1_noise1f = links.new(node_map1f.outputs['Vector'], node_noise1f.inputs['Vector'])  # Vector
    
    
    ## Create Voronoi Texture 
    # Properties
    node_vor1f = nodes.new(type='ShaderNodeTexVoronoi')   # Create node
    node_vor1f.feature = 'DISTANCE_TO_EDGE'          # Feature
    node_vor1f.voronoi_dimensions = '3D'      # Dimensions
    node_vor1f.location = curr_x1f, curr_y1f                # Location
    curr_x1f = curr_x1f + node_vor1f.width + space_w       # Update current x value
    
    # Inputs
    #print(node_vor.inputs.keys())
    node_vor1f.inputs['Scale'].default_value = (30 - 5) * percent + 5   # Scale
    node_vor1f.inputs['Randomness'].default_value = 0.0        # Randomness
    link1_vor1f = links.new(node_noise1f.outputs['Color'], node_vor1f.inputs['Vector'])  # Vector
    
    
    ## Create Color Ramp Node
    # Properties
    node_cr1f = nodes.new(type='ShaderNodeValToRGB')     # Create node
    colorRamp1f = node_cr1f.color_ramp                    # Color ramp
    cr1f_colors = [(1.0, 1.0, 1.0, 1), (0.0, 0.0, 0.0, 1)]   # Color ramp colors
    cr1f_secPos = (0.05 - 0.001) * percent + 0.001
    if (percent >= 0.5):
        cr1f_secPos = 0.01
    cr1f_pos = [0.0, cr1f_secPos]     # Color ramp positions
    updateColorRamp(colorRamp1f, cr1f_colors, cr1f_pos)    # Add pos and colors to color ramp
    colorRamp1f.color_mode = 'RGB'           # Color mode
    colorRamp1f.hue_interpolation = 'NEAR'   # Color interpolation
    colorRamp1f.interpolation = 'LINEAR'     # Interpolation between color stops
    node_cr1f.location = curr_x1f, curr_y1f         # Location
    curr_x1f = curr_x1f + node_cr1f.width + space_w       # Update current x value
    
    # Inputs
    link1_cr1f = links.new(node_vor1f.outputs['Distance'], node_cr1f.inputs['Fac'])  # Fac
    
    
    ## Bright / Contrast 1 for cracks
    # Properties
    node_bright1f = nodes.new(type='ShaderNodeBrightContrast')   # Create node
    node_bright1f.location = curr_x1f, curr_y1f                # Location
    
    # Inputs
    #print(node_vor.inputs.keys())
    node_bright1f.inputs['Bright'].default_value = 0.0           # Bright
    node_bright1f.inputs['Contrast'].default_value = 5.0        # Contrast
    link1_bright1f = links.new(node_cr1f.outputs['Color'], node_bright1f.inputs['Color'])  # Vector
    
    
    ## Bright / Contrast 2 for crack ends later
    # Properties
    node_bright2f = nodes.new(type='ShaderNodeBrightContrast')   # Create node
    node_bright2f.location = curr_x1f, curr_y1f - 2 * node_bright1f.height   # Location
    curr_x1f = curr_x1f + node_bright2f.width + space_w       # Update current x value
    
    # Inputs
    #print(node_vor.inputs.keys())
    node_bright2f.inputs['Bright'].default_value = 0.0           # Bright
    node_bright2f.inputs['Contrast'].default_value = 30.0        # Contrast
    link1_bright2f = links.new(node_cr1f.outputs['Color'], node_bright2f.inputs['Color'])  # Vector
    
    
    ### Make Cracks Have Ends
    curr_x2f = curr_x0f
    curr_y2f = curr_y1f - 2 * node_noise1f.height - space_w
    
    ## Create Noise Texture 2
    # Properties
    node_noise2f = nodes.new(type='ShaderNodeTexNoise')   # Create node
    node_noise2f.noise_dimensions = '3D'                  # Dimensions
    node_noise2f.location = curr_x2f, curr_y2f                       # Location
    curr_x2f = curr_x2f + node_noise2f.width + space_w       # Update current x value
    
    # Inputs
    #print(node_noise.inputs.keys())
    node_noise2f.inputs['Scale'].default_value = 10.0        # Scale
    node_noise2f.inputs['Detail'].default_value = 16.0       # Detail
    node_noise2f.inputs['Roughness'].default_value = 0.5     # Roughness
    node_noise2f.inputs['Distortion'].default_value = 0.16    # Distortion
    link1_noise2f = links.new(node_map1f.outputs['Vector'], node_noise2f.inputs['Vector'])  # Vector
    
    
    ## Create Color Ramp Node
    # Properties
    node_cr2f = nodes.new(type='ShaderNodeValToRGB')     # Create node
    colorRamp2f = node_cr2f.color_ramp                    # Color ramp
    cr2f_colors = [(0.0, 0.0, 0.0, 1), (1.0, 1.0, 1.0, 1)]   # Color ramp colors
    cr2f_pos = [0.4, 0.72]                           # Color ramp positions
    updateColorRamp(colorRamp2f, cr2f_colors, cr2f_pos)    # Add pos and colors to color ramp
    colorRamp2f.color_mode = 'RGB'           # Color mode
    colorRamp2f.hue_interpolation = 'NEAR'   # Color interpolation
    colorRamp2f.interpolation = 'LINEAR'     # Interpolation between color stops
    node_cr2f.location = curr_x2f, curr_y2f         # Location
    curr_x2f = curr_x2f + node_cr2f.width + space_w       # Update current x value
    
    # Inputs
    link1_cr2f = links.new(node_noise2f.outputs['Color'], node_cr2f.inputs['Fac'])  # Fac
    
    
    ### Connect all the crack stuff together
    curr_x3f = curr_x1f
    curr_y3f = curr_y1f
    
    ## mix RGB - Multiply
    # Properties
    node_mult1f = nodes.new(type='ShaderNodeMixRGB')   # Create node
    node_mult1f.blend_type = 'MULTIPLY'               # Operation
    node_mult1f.use_clamp = False                    # Use clamp
    node_mult1f.location = curr_x3f, curr_y3f                # Location
    
    # Inputs
    #print(node_vor.inputs.keys())
    node_mult1f.inputs['Fac'].default_value = 1.0           # Fac
    link1_mult1f = links.new(node_bright1f.outputs['Color'], node_mult1f.inputs['Color1'])  # Vector
    link2_mult1f = links.new(node_cr2f.outputs['Color'], node_mult1f.inputs['Color2'])  # Vector
    
    
    ## Mix RGB - Multiply
    # Properties
    node_mult2f = nodes.new(type='ShaderNodeMixRGB')   # Create node
    node_mult2f.blend_type = 'MULTIPLY'               # Operation
    node_mult2f.use_clamp = False                    # Use clamp
    node_mult2f.location = curr_x3f, curr_y3f - 2 * node_mult1f.height - space_w   # Location
    curr_x3f = curr_x3f + node_mult2f.width + space_w       # Update current x value
    
    # Inputs
    #print(node_vor.inputs.keys())
    node_mult2f.inputs['Fac'].default_value = 1.0           # Fac
    link1_mult2f = links.new(node_bright2f.outputs['Color'], node_mult2f.inputs['Color1'])  # Vector
    link2_mult2f = links.new(node_cr2f.outputs['Color'], node_mult2f.inputs['Color2'])  # Vector
    
    
    ## Invert Node
    # Properties
    node_inv1f = nodes.new(type='ShaderNodeInvert')   # Create node
    node_inv1f.location = curr_x3f, curr_y3f                # Location
    curr_x3f = curr_x3f + node_inv1f.width + space_w       # Update current x value
    
    # Inputs
    #print(node_vor.inputs.keys())
    node_inv1f.inputs['Fac'].default_value = 1.0           # Fac
    link1_inv1f = links.new(node_mult1f.outputs['Color'], node_inv1f.inputs['Color'])  # Vector
    



    ############################################
    """ Add Color, Cracks, and Spalling all together """
    
    curr_x0t = max(curr_x3c, curr_x1n, curr_x1s, curr_x3f) + space_w   # Find max width to start and then add a space
    
    ### Color
    curr_x1t = curr_x0t
    
    ## Create Mix Node 
    # Properties
    node_mix1t = nodes.new(type='ShaderNodeMixRGB')     # Create node
    node_mix1t.blend_type = 'MIX'                       # Blend Type
    node_mix1t.location = curr_x1t, curr_y2c         # Location
    curr_x1t = curr_x1t + node_mix1t.width + space_w       # Update current x value
    
    # Inputs
    link1_mix1t = links.new(node_cr1s.outputs['Color'], node_mix1t.inputs['Fac'])  # Fac
    link2_mix1t = links.new(node_rgbCurve1c.outputs['Color'], node_mix1t.inputs['Color1'])  # Color1
    link3_mix1t = links.new(node_rgbCurve2c.outputs['Color'], node_mix1t.inputs['Color2'])     # Color2
    
    
    ## Mix RGB - Multiply
    # Properties
    node_mult1t = nodes.new(type='ShaderNodeMixRGB')   # Create node
    node_mult1t.blend_type = 'MULTIPLY'               # Operation
    node_mult1t.use_clamp = False                    # Use clamp
    node_mult1t.location = curr_x1t, curr_y2c                # Location
    curr_x1t = curr_x1t + node_mult1t.width + space_w       # Update current x value
    
    # Inputs
    #print(node_vor.inputs.keys())
    node_mult1t.inputs['Fac'].default_value = 1.0           # Fac
    link1_mult1t = links.new(node_mix1t.outputs['Color'], node_mult1t.inputs['Color1'])  # Vector
    link2_mult1t = links.new(node_inv1f.outputs['Color'], node_mult1t.inputs['Color2'])  # Vector
    
    
    ## Mix RGB - Add
    # Properties
    node_mult2t = nodes.new(type='ShaderNodeMixRGB')   # Create node
    node_mult2t.blend_type = 'ADD'               # Operation
    node_mult2t.use_clamp = False                    # Use clamp
    node_mult2t.location = curr_x1t, curr_y2c                # Location
    curr_x1t = curr_x1t + node_mult2t.width + space_w       # Update current x value
    
    # Inputs
    #print(node_vor.inputs.keys())
    node_mult2t.inputs['Fac'].default_value = 0.25           # Fac
    link1_mult2t = links.new(node_cr1c.outputs['Color'], node_mult2t.inputs['Color1'])  # Vector
    link2_mult2t = links.new(node_mult1t.outputs['Color'], node_mult2t.inputs['Color2'])  # Vector
    
    
    
    ### Normal
    curr_x2t = curr_x0t
    
    ## Create Bump Node 1
    # Properties
    node_bump1t = nodes.new(type='ShaderNodeBump')     # Create node
    node_bump1t.invert = True                       # Invert bump map dir?
    node_bump1t.location = curr_x2t, curr_y1n       # Location
    curr_x2t = curr_x2t + node_bump1t.width + space_w       # Update current x value
    
    # Inputs
    #print(node_bump1.inputs.keys())
    node_bump1t.inputs['Strength'].default_value = 1.0        # Strength
    node_bump1t.inputs['Distance'].default_value = 1.0        # Distance
    link1_bump1t = links.new(node_mult2f.outputs['Color'], node_bump1t.inputs['Height'])  # Height
    link2_bump1t = links.new(node_normMap1n.outputs['Normal'], node_bump1t.inputs['Normal'])  # Normal
    
    
    ## Create Bump Node 2
    # Properties
    node_bump2t = nodes.new(type='ShaderNodeBump')     # Create node
    node_bump2t.invert = True                       # Invert bump map dir?
    node_bump2t.location = curr_x2t, curr_y1n       # Location
    curr_x2t = curr_x2t + node_bump2t.width + space_w       # Update current x value
    
    # Inputs
    #print(node_bump1.inputs.keys())
    node_bump2t.inputs['Strength'].default_value = 1.0        # Strength
    node_bump2t.inputs['Distance'].default_value = 1.0        # Distance
    link1_bump2t = links.new(node_cr1s.outputs['Color'], node_bump2t.inputs['Height'])  # Height
    link2_bump2t = links.new(node_bump1t.outputs['Normal'], node_bump2t.inputs['Normal'])  # Normal
    
    
    
    ### BSDF
    curr_x3t = max(curr_x1t, curr_x2t) + space_w
    
    ## Create Principled BSDF Node
    # Properties
    node_bsdf1t = nodes.new(type='ShaderNodeBsdfPrincipled')     # Create node
    node_bsdf1t.distribution = 'GGX'                       # Distribution
    node_bsdf1t.subsurface_method = 'BURLEY'               # Subsurface method
    node_bsdf1t.location = curr_x3t, curr_y1d         # Location
    curr_x3t = curr_x3t + node_bsdf1t.width + space_w       # Update current x value
    
    # Inputs
    link1_bsdf1t = links.new(node_mult2t.outputs['Color'], node_bsdf1t.inputs['Base Color'])  # Base color
    link2_bsdf1t = links.new(node_imgTex2c.outputs['Color'], node_bsdf1t.inputs['Roughness'])  # Base color
    link3_bsdf1t = links.new(node_bump2t.outputs['Normal'], node_bsdf1t.inputs['Normal'])  # Normal
    
    
    
    ### Displacement
    
    ## Vector Math
    # Properties
    node_vmath1t = nodes.new(type='ShaderNodeVectorMath')       # Create node
    node_vmath1t.operation = 'ADD'                                  # Operation
    node_vmath1t.location = curr_x3t, curr_y1d - 2 * node_bsdf1t.height - space_w   # Location
    curr_x3t = curr_x3t + node_vmath1t.width + space_w       # Update current x value
    
    # Inputs
    link1_vmath1t = links.new(node_math1d.outputs['Value'], node_vmath1t.inputs[0])  # Vector 1
    link2_vmath1t = links.new(node_mult2f.outputs['Color'], node_vmath1t.inputs[1])  # Vector 2
    
    
    ## Displacement
    # Properties
    node_disp1t = nodes.new(type='ShaderNodeDisplacement')       # Create node
    node_disp1t.space = 'OBJECT'                                  # Operation
    node_disp1t.location = curr_x3t, curr_y1d - 2 * node_bsdf1t.height - space_w   # Location
    curr_x3t = curr_x3t + node_disp1t.width + space_w       # Update current x value
    
    # Inputs
    link1_disp1t = links.new(node_vmath1t.outputs['Vector'], node_disp1t.inputs['Height'])  # Height
    node_disp1t.inputs['Midlevel'].default_value = 0.0                                      # Midlevel
    node_disp1t.inputs['Scale'].default_value = 100.0                                      # Scale
    
    
    
    ### Create Material Output Node
    # Properties
    node_matOut1t = nodes.new(type='ShaderNodeOutputMaterial')     # Create node
    node_matOut1t.is_active_output = True                       # Is active output?
    node_matOut1t.target = 'ALL'                       # Which renderer and viewport shading types to use the shaders for
    node_matOut1t.location = curr_x3t, curr_y1d         # Location
    curr_x3t = curr_x3t + node_matOut1t.width + space_w       # Update current x value
    
    # Inputs
    #print(node_bump1.inputs.keys())
    link1_matOut1t = links.new(node_bsdf1t.outputs['BSDF'], node_matOut1t.inputs['Surface'])  # Surface
    link2_matOut1t = links.new(node_disp1t.outputs['Displacement'], node_matOut1t.inputs['Displacement'])  # Displacement
    


############################################
############################################
############################################
""" Main Script """
# Inputs for material structural damage
drift_ratio = 0.5;      # Story Drift Ratio (unitless)
accel = 0;      # Peak Floor Acceleration in g
connHeight = 13.78;          # Top of wall height (ft)
strHeight = 68.9;           # Structure height above grade (ft)

# Plot fragility curves
#plotFragilityCurves()

# Folders
texFolder = 'textures' # Folder address to folder with textures
matImgName = 'Concrete024_8K'  # Material Image Name

print('Running')

# Create damaged materials
# createDamageMaterial(inOrOut, damagePara, connHeight, strHeight, texFolder, strImgName, matImgName)
createDamageMaterial(0, drift_ratio, connHeight, strHeight, texFolder, matImgName)    # Create in-plane damage material
#createDamageMaterial(1, accel, connHeight, strHeight, texFolder, matImgNam)    # Create out of plane damage material

print('Done')

