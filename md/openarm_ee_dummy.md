# OpenArm Tesollo: EE Dummy Link Setup

This note documents how to add end-effector (EE) dummy links for Isaac Lab
so they are discovered as bodies and match training body names.

## Goal
- Create palm-center EE links named `ll_dg_ee` and `rl_dg_ee`.
- Attach them to the Tesollo mount links with fixed joints.
- Ensure Isaac Lab `find_bodies()` can detect them (needs inertial + collision).
- Minimize physical side effects by using a tiny collision sphere.

## Where
- URDF: `/home/user/rl_ws/env_setup/openarm_tesollo.urdf`

## How
1) Add fixed joints from Tesollo mounts to EE links.
2) Add EE links with minimal inertial and tiny collision/visual geometry.

### Joint transforms (palm center offsets)
Left (tesollo base frame):
- translate: `x=0.023, y=0, z=0.13`
- rotate XYZ (rpy radians): `1.5707963 0 1.5707963`

Right (tesollo base frame):
- translate: `x=0.023, y=0, z=0.13`
- rotate XYZ (rpy radians): `1.5707963 3.1415927 1.5707963`

### Example URDF block
```xml
  <joint name="ll_dg_ee_joint" type="fixed">
    <origin xyz="0.023 0. 0.13" rpy="1.5707963 0. 1.5707963"/>
    <parent link="tesollo_left_ll_dg_mount"/>
    <child link="ll_dg_ee"/>
  </joint>
  <joint name="rl_dg_ee_joint" type="fixed">
    <origin xyz="0.023 0. 0.13" rpy="1.5707963 3.1415927 1.5707963"/>
    <parent link="tesollo_right_rl_dg_mount"/>
    <child link="rl_dg_ee"/>
  </joint>

  <link name="ll_dg_ee">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.001"/>
      <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.0001"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.0001"/>
      </geometry>
    </collision>
  </link>

  <link name="rl_dg_ee">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.001"/>
      <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.0001"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.0001"/>
      </geometry>
    </collision>
  </link>
```

## Notes
- `inertial` + `collision` ensures the link is recognized as a body.
- The tiny sphere keeps contact effects minimal.
- Keep the EE dummy inside the palm so it rarely contacts external objects.
- If names must match training configs, keep `ll_dg_ee`/`rl_dg_ee` as-is.
- If you want a pure “frame” with no contacts, remove `<collision>` blocks.

## Mesh Paths (relative)
Mesh paths in the URDF were converted to be relative to the URDF directory:
- Example: `meshes/openarm_left_link1_visual_mesh.obj`

This matches the layout:
- `/home/user/rl_ws/env_setup/openarm_tesollo.urdf`
- `/home/user/rl_ws/env_setup/meshes/`
