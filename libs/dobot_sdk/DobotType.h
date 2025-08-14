#ifndef DOBOTTYPE_H
#define DOBOTTYPE_H

#ifdef _MSC_VER
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef unsigned short uint16_t;
typedef signed short int16_t;
typedef unsigned int uint32_t;
typedef signed int int32_t;
typedef unsigned  long long uint64_t;
typedef signed long long int64_t;
#else
#include <stdint.h>
#endif

/*********************************************************************************************************
** Data structures
*********************************************************************************************************/

/*********************************************************************************************************
** Common parts
*********************************************************************************************************/
#pragma pack(push)
#pragma pack(1)

/*
 * Real-time pose
 */
typedef struct tagPose {
    float x;
    float y;
    float z;
    float r;
    float jointAngle[4];
}Pose;

/*
 * Kinematics parameters
 */
typedef struct tagKinematics {
    float velocity;
    float acceleration;
}Kinematics;

/*
 * HOME related
 */
typedef struct tagHOMEParams {
    float x;
    float y;
    float z;
    float r;
}HOMEParams;

typedef struct tagHOMECmd {
    uint32_t reserved;
}HOMECmd;

typedef struct tagAutoLevelingCmd {
    uint8_t controlFlag;
    float precision;
}AutoLevelingCmd;

/*
 * Hand hold teach
 */
typedef enum tagHHTTrigMode {
    TriggeredOnKeyReleased,
    TriggeredOnPeriodicInterval
}HHTTrigMode;

/*
 * End effector
 */
typedef struct tagEndEffectorParams {
    float xBias;
    float yBias;
    float zBias;
}EndEffectorParams;

/*
 * Arm orientation
 */
typedef enum tagArmOrientation {
    LeftyArmOrientation,
    RightyArmOrientation,
}ArmOrientation;

/*
 * JOG related
 */
typedef struct tagJOGJointParams {
    float velocity[4];
    float acceleration[4];
}JOGJointParams;

typedef struct tagJOGCoordinateParams {
    float velocity[4];
    float acceleration[4];
}JOGCoordinateParams;

typedef struct tagJOGLParams {
    float velocity;
    float acceleration;
}JOGLParams;

typedef struct tagJOGCommonParams {
    float velocityRatio;
    float accelerationRatio;
}JOGCommonParams;

enum {
    JogIdle,
    JogAPPressed,
    JogANPressed,
    JogBPPressed,
    JogBNPressed,
    JogCPPressed,
    JogCNPressed,
    JogDPPressed,
    JogDNPressed,
    JogEPPressed,
    JogENPressed
};

typedef struct tagJOGCmd {
    uint8_t isJoint;
    uint8_t cmd;
}JOGCmd;

/*
 * PTP related
 */
typedef struct tagPTPJointParams {
    float velocity[4];
    float acceleration[4];
}PTPJointParams;

typedef struct tagPTPCoordinateParams {
    float xyzVelocity;
    float rVelocity;
    float xyzAcceleration;
    float rAcceleration;
}PTPCoordinateParams;

typedef struct tagPTPLParams {
    float velocity;
    float acceleration;
}PTPLParams;

typedef struct tagPTPJumpParams {
    float jumpHeight;
    float zLimit;
}PTPJumpParams;

typedef struct tagPTPJump2Params {
    float startJumpHeight;
    float endJumpHeight;
    float zLimit;
}PTPJump2Params;

typedef struct tagPTPCommonParams {
    float velocityRatio;
    float accelerationRatio;
}PTPCommonParams;

enum PTPMode {
    PTPJUMPXYZMode,
    PTPMOVJXYZMode,
    PTPMOVLXYZMode,

    PTPJUMPANGLEMode,
    PTPMOVJANGLEMode,
    PTPMOVLANGLEMode,

    PTPMOVJANGLEINCMode,
    PTPMOVLXYZINCMode,
    PTPMOVJXYZINCMode,

    PTPJUMPMOVLXYZMode,
};

typedef struct tagPTPCmd {
    uint8_t ptpMode;
    float x;
    float y;
    float z;
    float r;
}PTPCmd;

typedef struct tagPTPWithLCmd {
    uint8_t ptpMode;
    float x;
    float y;
    float z;
    float r;
    float l;
}PTPWithLCmd;

typedef struct tagParallelOutputCmd {
    uint8_t ratio;
    uint16_t address;
    uint8_t level;
}ParallelOutputCmd;

/*
 * CP related
 */
typedef struct tagCPParams
{
    float planAcc;
    float juncitionVel;
    union {
        float acc;
        float period;
    };
    uint8_t realTimeTrack;
}CPParams;

typedef struct tagCPCommonParams {
    float velocityRatio;
    float accelerationRatio;
}CPCommonParams;

enum CPMode {
    CPRelativeMode,
    CPAbsoluteMode
};

typedef struct tagCPCmd {
    uint8_t cpMode;
    float x;
    float y;
    float z;
    union {
        float velocity;
        float power;
    };
}CPCmd;

/*
 * ARC related
 */
typedef struct tagARCParams {
    float xyzVelocity;
    float rVelocity;
    float xyzAcceleration;
    float rAcceleration;
}ARCParams;

typedef struct tagARCCommonParams {
    float velocityRatio;
    float accelerationRatio;
}ARCCommonParams;

typedef struct tagARCCmd {
    struct {
        float x;
        float y;
        float z;
        float r;
    }cirPoint;
    struct {
        float x;
        float y;
        float z;
        float r;
    }toPoint;
}ARCCmd;

typedef struct tagCircleCmd {
    struct {
        float x;
        float y;
        float z;
        float r;
    }cirPoint;
    struct {
        float x;
        float y;
        float z;
        float r;
    }toPoint;
    uint32_t count;
}CircleCmd;

typedef struct tagWAITCmd {
    uint32_t timeout;
}WAITCmd;

typedef enum tagTRIGMode {
    TRIGInputIOMode,
    TRIGADCMode
}TRIGMode;

typedef enum tagTRIGInputIOCondition {
    TRIGInputIOEqual,
    TRIGInputIONotEqual
}TRIGInputIOCondition;

typedef enum tagTRIGADCCondition {
    TRIGADCLT,  // Lower than
    TRIGADCLE,  // Lower than or Equal
    TRIGADCGE,  // Greater than or Equal
    TRIGADCGT   // Greater Than
}TRIGADCCondition;

typedef struct tagTRIGCmd {
    uint8_t address;
    uint8_t mode;
    uint8_t condition;
    uint16_t threshold;
}TRIGCmd;

typedef enum tagIOFunction {
    IOFunctionDummy,
    IOFunctionDO,
    IOFunctionPWM,
    IOFunctionDI,
    IOFunctionADC
}IOFunction;

typedef struct tagIOMultiplexing {
    uint8_t address;
    uint8_t multiplex;
}IOMultiplexing;

typedef struct tagIODO {
    uint8_t address;
    uint8_t level;
}IODO;

typedef struct tagIOPWM {
    uint8_t address;
    float frequency;
    float dutyCycle;
}IOPWM;

typedef struct tagIODI {
    uint8_t address;
    uint8_t level;
}IODI;

typedef struct tagIOADC {
    uint8_t address;
    uint16_t value;
}IOADC;

typedef struct tagEMotor {
    uint8_t index;
    uint8_t isEnabled;
    int32_t speed;
}EMotor;

typedef struct tagEMotorS {
    uint8_t index;
    uint8_t isEnabled;
    int32_t speed;
    uint32_t distance;
}EMotorS;

/*
 * WIFI related
 */
typedef struct tagWIFIIPAddress {
    uint8_t dhcp;
    uint8_t addr[4];
}WIFIIPAddress;

typedef struct tagWIFINetmask {
    uint8_t addr[4];
}WIFINetmask;

typedef struct tagWIFIGateway {
    uint8_t addr[4];
}WIFIGateway;

typedef struct tagWIFIDNS {
    uint8_t addr[4];
}WIFIDNS;

/*
 * Test
 */
typedef struct tagUserParams{
    float params[8];
}UserParams;

/*
 * Firmware related
 */

enum FirmwareSwitchMode{
    NO_SWITCH,
    DOBOT_SWITCH,
    PRINTING_SWITCH,
    DRIVER1_SWITCH,
    DRIVER2_SWITCH,
    DRIVER3_SWITCH,
    DRIVER4_SWITCH,
    DRIVER5_SWITCH
};

typedef struct tagFirmwareParams {
    uint8_t  mode;
}FirmwareParams;

enum FirewareMode{
    INVALID_MODE,
    DOBOT_MODE,
    PRINTING_MODE,
    OFFLINE_MODE
};

typedef struct tagFirmwareModes {
    uint8_t  mode;
    uint8_t  ctl; //0 or 1
}FirmwareMode;

typedef enum tagServoControlLoop{
    ServoPositionLoop,
    ServoVelocityLoop,
    ServoCurrentLoop
}ServoControlLoop;

typedef struct tagPIDParams{
    float p;
    float i;
    float d;
    float v;
    float a;
}PIDParams;

typedef struct tagPID{
    uint8_t index;
    uint8_t controlLoop;
    PIDParams params;
}PID;

typedef enum tagColorPort{
    CL_PORT_GP1,
    CL_PORT_GP2,
    CL_PORT_GP4,
    CL_PORT_GP5
}ColorPort;

typedef enum tagInfraredPort{
    IF_PORT_GP1,
    IF_PORT_GP2,
    IF_PORT_GP4,
    IF_PORT_GP5
}InfraredPort;

typedef enum tagUART4PeripheralsType {
    UART4PeripheralsUART,
    UART4PeripheralsWIFI,
    UART4PeripheralsBLE,
    UART4PeripheralsCH375
} UART4PeripheralsType;

typedef struct tagPluseCmd {
    float j1;
    float j2;
    float j3;
    float j4;
    float e1;
    float e2;
} PluseCmd;

/*********************************************************************************************************
** API result
*********************************************************************************************************/
enum {
    DobotConnect_NoError,
    DobotConnect_NotFound,
    DobotConnect_Occupied
};

enum {
    DobotCommunicate_NoError,
    DobotCommunicate_BufferFull,
    DobotCommunicate_Timeout,
    DobotCommunicate_InvalidParams
};

#pragma pack(pop)
#endif
