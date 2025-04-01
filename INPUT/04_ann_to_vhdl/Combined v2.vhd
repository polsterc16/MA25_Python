-- VHDL Entity {$NAME_ENTITY}
--
-- Created: {$DATE_TIME}
--
--
LIBRARY ieee;
USE ieee.std_logic_1164.all;
USE ieee.numeric_std.all;

entity {$NAME_ENTITY} is
   port( 
      clk          : in     std_logic;
      reset        : in     std_logic;
      
      src_TX      : in     std_logic;
      ack_RX      : out    std_logic;
      
      dst_RX      : in     std_logic;
      ready_to_TX : out    std_logic;
      
      -- layer_in    : in     t_array_data_stdlv (0 to 1);
      -- layer_out   : out    t_array_data_stdlv (0 to 1)
{$PORT_LAYER_IN_OUT}
   );

-- Declarations

end {$NAME_ENTITY} ;

--
-- VHDL Architecture {$NAME_ENTITY}.struct
--
-- Created: {$DATE_TIME}
--
--
LIBRARY ieee;
USE ieee.std_logic_1164.all;
USE ieee.numeric_std.all;
library proj_master_2025_lib;
use proj_master_2025_lib.p_002_generic_01.all;


architecture struct of {$NAME_ENTITY} is

   -- Architecture declarations

   -- Internal signal declarations
   -- signal dst_RX1      : std_logic;
   -- signal layer_out1   : t_array_data_stdlv(0 to 1);
   -- signal ready_to_TX1 : std_logic;
{$SIGNAL_DECLARATION}


   -- Component Declarations
   component c_004_layer_01
   generic (
      g_layer_length_cur  : integer               := 4;
      g_layer_length_prev : integer               := 2;
      g_layer_bias        : t_array_integer       := (0,0,0,0);
      g_layer_weights     : t_array2D_integer     := ((0,0),(0,0),(0,0),(0,0));
      g_act_func          : t_activation_function := AF_RELU
   );
   port (
      clk         : in     std_logic ;
      reset       : in     std_logic ;
      dst_RX      : in     std_logic ;
      src_TX      : in     std_logic ;
      ready_to_TX : out    std_logic ;
      ack_RX      : out    std_logic ;
      layer_in    : in     t_array_data_stdlv (0 to g_layer_length_prev-1);
      layer_out   : out    t_array_data_stdlv (0 to g_layer_length_cur-1)
   );
   end component;

   -- Optional embedded configurations
   -- pragma synthesis_off
   for all : c_004_layer_01 use entity proj_master_2025_lib.c_004_layer_01;
   -- pragma synthesis_on


begin

   -- Instance port mappings.
{$INSTANCE_PORT_MAPPINGS}

end struct;
