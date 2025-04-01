-- VHDL Entity c_x_ANN_01
--
-- Created: 2025-04-01 17:47:37.940914
--
--
LIBRARY ieee;
USE ieee.std_logic_1164.all;
USE ieee.numeric_std.all;

entity c_x_ANN_01 is
   port( 
      clk          : in     std_logic;
      reset        : in     std_logic;
      
      src_TX      : in     std_logic;
      ack_RX      : out    std_logic;
      
      dst_RX      : in     std_logic;
      ready_to_TX : out    std_logic;
      
      -- layer_in    : in     t_array_data_stdlv (0 to 1);
      -- layer_out   : out    t_array_data_stdlv (0 to 1)
layer_in : in t_array_data_stdlv (0 to 1);
layer_out : out t_array_data_stdlv (0 to 0)
   );

-- Declarations

end c_x_ANN_01 ;

--
-- VHDL Architecture c_x_ANN_01.struct
--
-- Created: 2025-04-01 17:47:37.940914
--
--
LIBRARY ieee;
USE ieee.std_logic_1164.all;
USE ieee.numeric_std.all;
library proj_master_2025_lib;
use proj_master_2025_lib.p_002_generic_01.all;


architecture struct of c_x_ANN_01 is

   -- Architecture declarations

   -- Internal signal declarations
   -- signal dst_RX1      : std_logic;
   -- signal layer_out1   : t_array_data_stdlv(0 to 1);
   -- signal ready_to_TX1 : std_logic;
signal DST_RX_0 : std_logic;
signal R2TX_0 : std_logic;
signal layer_0 : t_array_data_stdlv(0 to 7);


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
U_0 : c_004_layer_01
generic map (
  g_layer_length_cur => 8,
  g_layer_length_prev => 2,
  g_layer_bias => ( -46, -277, 1097, -357, -281, -252, -261, -466 ),
  g_layer_weights => ( (891, -462), (-841, -530), (143, 40), (850, 336), (57, 989), (-901, -168), (-357, 869), (198, -1016) ),
  g_act_func => AF_RELU
)
port map (
  clk => clk,
  reset => reset,
  ack_RX => ack_RX,
  src_TX => src_TX,
  layer_in => layer_in,
  dst_RX => DST_RX_0,
  ready_to_TX => R2TX_0,
  layer_out => layer_0
);

U_1 : c_004_layer_01
generic map (
  g_layer_length_cur => 1,
  g_layer_length_prev => 8,
  g_layer_bias => ( 0 => 946 ),
  g_layer_weights => ( 0 => (-1101, -1042, 1167, -1578, -1099, -1005, -1222, -1404) ),
  g_act_func => AF_HARD_SIGMOID
)
port map (
  clk => clk,
  reset => reset,
  ack_RX => DST_RX_0,
  src_TX => R2TX_0,
  layer_in => layer_0,
  dst_RX => dst_RX,
  ready_to_TX => ready_to_TX,
  layer_out => layer_out
);


end struct;
