from keras.models import Model
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose


def unet3d_res(img_rows, img_cols, img_chns, img_depth, pretrained_weight_path=False, **kwargs):
    inputs = Input((img_depth, img_rows, img_cols, img_chns))
    conv11 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', name='conv11')(inputs)  # (16, 256, 256, 32)
    conc11 = concatenate([inputs, conv11], axis=4, name='conc11')  # (16, 256, 256, 33)
    conv12 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', name='conv12')(conc11)  # (16, 256, 256, 32)
    conc12 = concatenate([inputs, conv12], axis=4, name='conc12')  # (16, 256, 256, 33)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2), name='pool1')(conc12)  # (8, 128, 128, 33)

    conv21 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='conv21')(pool1)  # (8, 128, 128, 64)
    conc21 = concatenate([pool1, conv21], axis=4, name='conc21')  # (8, 128, 128, 97)
    conv22 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='conv22')(conc21)  # (8, 128, 128, 64)
    conc22 = concatenate([pool1, conv22], axis=4, name='conc22')  # (8, 128, 128, 97)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), name='MaxPool3D_2')(conc22)  # (4, 64, 64, 97)

    conv31 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='conv31')(pool2)  # (4, 64, 64, 128)
    conc31 = concatenate([pool2, conv31], axis=4, name='conc31')  # (4, 64, 64, 225)
    conv32 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='conv32')(conc31)  # (4, 64, 64, 128)
    conc32 = concatenate([pool2, conv32], axis=4, name='conc32')  # (4, 64, 64, 225)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2), name='MaxPool3D_3')(conc32)  # (2, 32, 32, 225)

    conv41 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='conv41')(pool3)  # (2, 32, 32, 256)
    conc41 = concatenate([pool3, conv41], axis=4, name='conc41')  # (2, 32, 32, 481)
    conv42 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='conv42')(conc41)  # (2, 32, 32, 256)
    conc42 = concatenate([pool3, conv42], axis=4, name='conc42')  # (2, 32, 32, 481)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2), name='MaxPool3D_4')(conc42)  # (1, 16, 16, 481)

    conv51 = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='conv51')(pool4)  # (1, 16, 16, 512)
    conc51 = concatenate([pool4, conv51], axis=4, name='conc51')  # (1, 16, 16, 993)
    conv52 = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='conv52')(conc51)  # (1, 16, 16, 512)
    conc52 = concatenate([pool4, conv52], axis=4, name='conc52')  # (1, 16, 16, 993)

    conv_tr_6 = Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same', name='conv_tr_6')(conc52)  # (2, 32, 32, 256)
    conc60 = concatenate([conv_tr_6, conv42], axis=4, name='conc60')  # (2, 32, 32, 512)

    conv61 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='conv61')(conc60)  # (2, 32, 32, 256)
    conc61 = concatenate([conc60, conv61], axis=4, name='conc61')  # (2, 32, 32, 768)
    conv62 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='conv62')(conc61)  # (2, 32, 32, 256)
    conc62 = concatenate([conc60, conv62], axis=4, name='conc62')  # (2, 32, 32, 768)

    conv_tr_7 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same', name='conv_tr_7')(conc62)  # (4, 64, 64, 128)
    conc70 = concatenate([conv_tr_7, conv32], axis=4)  # (4, 64, 64, 256)

    conv71 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='conv71')(conc70)  # (4, 64, 64, 128)
    conc71 = concatenate([conc70, conv71], axis=4, name='conc71')  # (4, 64, 64, 384)
    conv72 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='conv72')(conc71)  # (4, 64, 64, 128)
    conc72 = concatenate([conc70, conv72], axis=4, name='conc72')  # (4, 64, 64, 384)

    conv_tr_8 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc72)  # (8, 128, 128, 64)
    conc80 = concatenate([conv_tr_8, conv22], axis=4)  # (8, 128, 128, 128)

    conv81 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='conv81')(conc80)  # (8, 128, 128, 64)
    conc81 = concatenate([conc80, conv81], axis=4, name='conc81')  # (8, 128, 128, 192)
    conv82 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='conv82')(conc81)  # (8, 128, 128, 64)
    conc82 = concatenate([conc80, conv82], axis=4, name='conc82')  # (8, 128, 128, 192)

    conv_tr_9 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc82)  # (16, 256, 256, 32)
    conc90 = concatenate([conv_tr_9, conv12], axis=4)  # (16, 256, 256, 64)

    conv91 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', name='conv91')(conc90)  # (16, 256, 256, 32)
    conc91 = concatenate([conc90, conv91], axis=4, name='conc91')  # (16, 256, 256, 96)
    conv92 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', name='conv92')(conc91)  # (16, 256, 256, 32)
    conc92 = concatenate([conc90, conv92], axis=4, name='conc92')  # (16, 256, 256, 96)

    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid', name='conv10')(conc92)  # (16, 256, 256, 1)

    model = Model(inputs=[inputs], outputs=[conv10], name='Unet_Res_32x512')
    if pretrained_weight_path:
        model.load_weights(pretrained_weight_path)
    return model


if __name__ == '__main__':
    model_class_obj = unet3d_res()
    model_class_obj.summary(line_length=140)
