
import sys
sys.path.append('../')
from PlotNeuralNet.pycore.tikzeng import *
from PlotNeuralNet.pycore.blocks import *

arch = [ 
    to_head('..'), 
    to_cor(),
    to_begin(),
    
    #input
    to_input("021.png", to="(-3.3,0,0)", width=8, height=6),

    #block-001
    to_ConvConvRelu( name='ccr_b1', s_filer=480, n_filer=(32,32), offset="(0,0,0)", to="(0,0,0)", width=(2,2), height=32, depth=40  ),
    to_Pool(name="pool_b1", offset="(0,0,0)", to="(ccr_b1-east)", width=1, height=24, depth=32, opacity=0.5),
    
    *block_2ConvPool( name='b2', botton='pool_b1', top='pool_b2', s_filer=240, n_filer=64, offset="(1,0,0)", size=(24,32,3.5), opacity=0.5 ),
    *block_2ConvPool( name='b3', botton='pool_b2', top='pool_b3', s_filer=120,  n_filer=128, offset="(1,0,0)", size=(18,24,5.5), opacity=0.5 ),

    #Bottleneck
    #block-005
    to_ConvConvRelu( name='ccr_b4', s_filer=60, n_filer=(256,256), offset="(2,0,0)", to="(pool_b3-east)", width=(8,8), height=12, depth=16, caption="Bottleneck"  ),
    to_connection( "pool_b3", "ccr_b4"),

    #Decoder
    *block_UnconvUNet( name="b5", botton="ccr_b4", top='end_b5', s_filer=120,  n_filer=128, offset="(2.1,0,0)", size=(18,24,5.0), opacity=0.5 ),
    to_skip( of='ccr_b3', to='ccr_rec_b5', pos=1.3),
    *block_UnconvUNet( name="b6", botton="end_b5", top='end_b6', s_filer=240, n_filer=64, offset="(2.1,0,0)", size=(24,32,3.5), opacity=0.5 ),
    to_skip( of='ccr_b2', to='ccr_rec_b6', pos=1.3),
    
    *block_UnconvUNet( name="b7", botton="end_b6", top='end_b7', s_filer=480, n_filer=32,  offset="(2.1,0,0)", size=(32,40,2.5), opacity=0.5 ),
    to_skip( of='ccr_b1', to='ccr_rec_b7', pos=1.3),

    to_Conv(name="b8", s_filer=480,
            offset = "(0.75,0,0)", to = "(end_b7-east)", n_filer = 2, width = 1.5, height = 32, depth = 40),
    to_connection( "end_b7", "b8"),
    
    to_ConvSoftMax( name="sigmoid1", s_filer=480, offset="(2,0,0)", to="(end_b7-east)", width=1, height=32, depth=40, caption="SIGMOID" ),
    to_connection( "b8", "sigmoid1"),
    # output image workaround
    to_input( '021_predict.png',to='(31.7,0,0)', width=8, height=6 ),
    to_end() 
    ]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
    
