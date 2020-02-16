// Generated by gencpp from file lgsvl_msgs/BoundingBox3D.msg
// DO NOT EDIT!


#ifndef LGSVL_MSGS_MESSAGE_BOUNDINGBOX3D_H
#define LGSVL_MSGS_MESSAGE_BOUNDINGBOX3D_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Vector3.h>

namespace lgsvl_msgs
{
template <class ContainerAllocator>
struct BoundingBox3D_
{
  typedef BoundingBox3D_<ContainerAllocator> Type;

  BoundingBox3D_()
    : position()
    , size()  {
    }
  BoundingBox3D_(const ContainerAllocator& _alloc)
    : position(_alloc)
    , size(_alloc)  {
  (void)_alloc;
    }



   typedef  ::geometry_msgs::Pose_<ContainerAllocator>  _position_type;
  _position_type position;

   typedef  ::geometry_msgs::Vector3_<ContainerAllocator>  _size_type;
  _size_type size;





  typedef boost::shared_ptr< ::lgsvl_msgs::BoundingBox3D_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::lgsvl_msgs::BoundingBox3D_<ContainerAllocator> const> ConstPtr;

}; // struct BoundingBox3D_

typedef ::lgsvl_msgs::BoundingBox3D_<std::allocator<void> > BoundingBox3D;

typedef boost::shared_ptr< ::lgsvl_msgs::BoundingBox3D > BoundingBox3DPtr;
typedef boost::shared_ptr< ::lgsvl_msgs::BoundingBox3D const> BoundingBox3DConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::lgsvl_msgs::BoundingBox3D_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::lgsvl_msgs::BoundingBox3D_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace lgsvl_msgs

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': True, 'IsMessage': True, 'HasHeader': False}
// {'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg'], 'sensor_msgs': ['/opt/ros/kinetic/share/sensor_msgs/cmake/../msg'], 'geometry_msgs': ['/opt/ros/kinetic/share/geometry_msgs/cmake/../msg'], 'lgsvl_msgs': ['/home/deepaktalwardt/Dropbox/SJSU/Semesters/Fall2019/CMPE256/Project/ros_workspace/src/lgsvl_msgs/msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::lgsvl_msgs::BoundingBox3D_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::lgsvl_msgs::BoundingBox3D_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::lgsvl_msgs::BoundingBox3D_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::lgsvl_msgs::BoundingBox3D_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::lgsvl_msgs::BoundingBox3D_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::lgsvl_msgs::BoundingBox3D_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::lgsvl_msgs::BoundingBox3D_<ContainerAllocator> >
{
  static const char* value()
  {
    return "0afc39928ba33aad299a6acabb48fd7d";
  }

  static const char* value(const ::lgsvl_msgs::BoundingBox3D_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x0afc39928ba33aadULL;
  static const uint64_t static_value2 = 0x299a6acabb48fd7dULL;
};

template<class ContainerAllocator>
struct DataType< ::lgsvl_msgs::BoundingBox3D_<ContainerAllocator> >
{
  static const char* value()
  {
    return "lgsvl_msgs/BoundingBox3D";
  }

  static const char* value(const ::lgsvl_msgs::BoundingBox3D_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::lgsvl_msgs::BoundingBox3D_<ContainerAllocator> >
{
  static const char* value()
  {
    return "geometry_msgs/Pose position  # 3D position and orientation of the bounding box center in Lidar space, in meters\n\
geometry_msgs/Vector3 size  # Size of the bounding box, in meters\n\
\n\
================================================================================\n\
MSG: geometry_msgs/Pose\n\
# A representation of pose in free space, composed of position and orientation. \n\
Point position\n\
Quaternion orientation\n\
\n\
================================================================================\n\
MSG: geometry_msgs/Point\n\
# This contains the position of a point in free space\n\
float64 x\n\
float64 y\n\
float64 z\n\
\n\
================================================================================\n\
MSG: geometry_msgs/Quaternion\n\
# This represents an orientation in free space in quaternion form.\n\
\n\
float64 x\n\
float64 y\n\
float64 z\n\
float64 w\n\
\n\
================================================================================\n\
MSG: geometry_msgs/Vector3\n\
# This represents a vector in free space. \n\
# It is only meant to represent a direction. Therefore, it does not\n\
# make sense to apply a translation to it (e.g., when applying a \n\
# generic rigid transformation to a Vector3, tf2 will only apply the\n\
# rotation). If you want your data to be translatable too, use the\n\
# geometry_msgs/Point message instead.\n\
\n\
float64 x\n\
float64 y\n\
float64 z\n\
";
  }

  static const char* value(const ::lgsvl_msgs::BoundingBox3D_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::lgsvl_msgs::BoundingBox3D_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.position);
      stream.next(m.size);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct BoundingBox3D_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::lgsvl_msgs::BoundingBox3D_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::lgsvl_msgs::BoundingBox3D_<ContainerAllocator>& v)
  {
    s << indent << "position: ";
    s << std::endl;
    Printer< ::geometry_msgs::Pose_<ContainerAllocator> >::stream(s, indent + "  ", v.position);
    s << indent << "size: ";
    s << std::endl;
    Printer< ::geometry_msgs::Vector3_<ContainerAllocator> >::stream(s, indent + "  ", v.size);
  }
};

} // namespace message_operations
} // namespace ros

#endif // LGSVL_MSGS_MESSAGE_BOUNDINGBOX3D_H