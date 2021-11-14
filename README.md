# Henry's Laser Turret

## Jetson Setup

Make sure ROS Melodic is installed

Make a workspace

        mkdir -p ~/catkin_ws/src
        cd ~/catkin_ws/src
        git clone --recursive https://github.com/davesarmoury/HenryLaserTurret.git
        cd ~/catkin_ws
        rosdep install --from-paths src --ignore-src -yr
        catkin_make

*Make sure your hosts are setup properly in `/etc/hosts`*

Add these lines to your `.bashrc`

        source ~/catkin_ws/devel/setup.bash
        export ROS_MASTER_URI=http://<ned_hostname>:11311
 
Install chrony

        sudo apt install chrony

Add this line to the bottom of cat /etc/chrony/chrony.conf 

        allow 192.168.2.0/24

Restart chrony (or the whole machine)

## Ned Setup

SSH into ned

        touch ~/catkin_ws/src/niryo_robot_description/CATKIN_IGNORE
        touch ~/catkin_ws/src/niryo_robot_msgs/CATKIN_IGNORE

        cd  ~/catkin_ws/src/
        git clone --recursive https://github.com/davesarmoury/HenryLaserTurret.git

        cd  ~/catkin_ws/
        rm -rf build devel

        catkin_make

*Make sure your hosts are setup properly in `/etc/hosts`*

Disable the firewall

        sudo ufw disable

Update Networking. Change `/etc/network/interfaces` to look like below

        auto lo
        iface lo inet loopback

        #allow-hotplug eth0
        #iface eth0 inet static
        #    address 169.254.200.200
        #    netmask 255.255.0.0

        allow-hotplug eth0
        iface eth0 inet dhcp

Install chrony

        sudo apt install chrony

Add this line to the top of cat /etc/chrony/chrony.conf

        server <Jetson Hostname> iburst prefer

Restart chrony (or the whole machine)




