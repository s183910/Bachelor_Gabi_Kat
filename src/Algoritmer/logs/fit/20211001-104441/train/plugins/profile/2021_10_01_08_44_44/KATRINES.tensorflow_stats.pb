"?D
DDeviceIDLE"IDLE1?????ƠBA?????ƠBQ      ??Y      ???Unknown
BHostIDLE"IDLE1+?9պ@A+?9պ@aÁ??<???iÁ??<????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1???a??@9???a??@A???a??@I???a??@ai ?????i??6?@)???Unknown?
?HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1`??"???@9`??"???@A`??"???@I`??"???@aULI_???i????K???Unknown
oHost_FusedMatMul"sequential/dense/Relu(1?I+,?@9?I+,?@A?I+,?@I?I+,?@al???^??i4/0hW???Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1Zd;?O??@9Zd;?O??@AZd;?O??@IZd;?O??@a2?ɫ?iT??y????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1?VN?@9?VN?@A?VN?@I?VN?@aO&?X???i?t"N???Unknown
^HostGatherV2"GatherV2(1???Mb?d@9???Mb?d@A???Mb?d@I???Mb?d@a-!??Q??i>??zß???Unknown
}	HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1㥛? xb@9㥛? xb@A㥛? xb@I㥛? xb@a߮'f???i?5I????Unknown
{
HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1`??"?`@9`??"?`@A`??"?`@I`??"?`@a??j?4?i+~
b?%???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(19??v?/Y@99??v?/Y@A9??v?/Y@I9??v?/Y@a?7#?1bx?i????vV???Unknown
tHost_FusedMatMul"sequential/dense_1/BiasAdd(1???Mb@U@9???Mb@U@A???Mb@U@I???Mb@U@a??????t?i]???????Unknown
oHostMul"sequential/dropout/dropout/Mul(1?V?T@9?V?T@A?V?T@I?V?T@a?'"???s?i?>6?W????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1ˡE???R@9ˡE???R@AˡE???R@IˡE???R@aQm???br?i???????Unknown
rHostDataset"Iterator::Root::ParallelMapV2(1?5^?I?I@9?5^?I?I@A?5^?I?I@I?5^?I?I@a???7(i?i1? ?E????Unknown
?HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(11?Z4H@91?Z4H@A1?Z4H@I1?Z4H@a}?'5?ng?i?6δ????Unknown
?Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1NbX9?G@9NbX9?G@ANbX9?G@INbX9?G@aɦq?M?f?i|?y???Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1?$??E@9?$??E@A?$??E@I?$??E@a?,UNXd?i??Y2?'???Unknown
?HostDataset"4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat(1?ZdkB@9?ZdkB@A??(\?b?@I??(\?b?@a????lb^?iv`?h7???Unknown
?HostDataset".Iterator::Root::ParallelMapV2::Zip[0]::FlatMap(1m????RC@9m????RC@A???S?%?@I???S?%?@aaw?Я'^?i2ϛ@F???Unknown
oHostSoftmax"sequential/dense_1/Softmax(1???(\?;@9???(\?;@A???(\?;@I???(\?;@aġ8?f?Z?i???s?S???Unknown
qHostCast"sequential/dropout/dropout/Cast(1u?V?:@9u?V?:@Au?V?:@Iu?V?:@aG?$F??Y?i?? Bg`???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1??? ??9@9??? ??9@A??? ??9@I??? ??9@a?????X?ij??:?l???Unknown
iHostWriteSummary"WriteSummary(1??/?$9@9??/?$9@A??/?$9@I??/?$9@a?jWq?WX?i??y???Unknown?
cHostDataset"Iterator::Root(1㥛? HQ@9㥛? HQ@A-???'1@I-???'1@a?@k???P?i???a????Unknown
?HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1/?$??+@9/?$??+@A/?$??+@I/?$??+@a	!9?c?J?i??????Unknown
qHostMul" sequential/dropout/dropout/Mul_1(1?? ?r?*@9?? ?r?*@A?? ?r?*@I?? ?r?*@a1????I?i~??ׁ????Unknown
?HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1h??|?5(@9h??|?5(@Ah??|?5(@Ih??|?5(@aP?tE?oG?i?(??]????Unknown
~HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1??? ??'@9??? ??'@A??? ??'@I??? ??'@aG??/G?i0]i?)????Unknown
eHost
LogicalAnd"
LogicalAnd(1???Qx&@9???Qx&@A???Qx&@I???Qx&@a?}???E?iq<?ϙ????Unknown?
HostMul".gradient_tape/sequential/dropout/dropout/Mul_2(1? ?rhQ%@9? ?rhQ%@A? ?rhQ%@I? ?rhQ%@a`=Rn?D?i $?¤???Unknown
[ HostAddV2"Adam/add(1T㥛? %@9T㥛? %@AT㥛? %@IT㥛? %@a??<WtD?i>??ߩ???Unknown
w!HostDataset""Iterator::Root::ParallelMapV2::Zip(1Zd;?OMU@9Zd;?OMU@A??x?&q#@I??x?&q#@a?"?{?B?i??_?????Unknown
Z"HostArgMax"ArgMax(1{?G?:"@9{?G?:"@A{?G?:"@I{?G?:"@aLw???A?i,????????Unknown
?#HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1J+?"@9J+?"@AJ+?"@IJ+?"@aaj???A?i?\ؠ^????Unknown
l$HostIteratorGetNext"IteratorGetNext(1??C?l?!@9??C?l?!@A??C?l?!@I??C?l?!@a?-YA?iL?#w?????Unknown
v%HostAssignAddVariableOp"AssignAddVariableOp_2(1Zd;?? @9Zd;?? @AZd;?? @IZd;?? @a?A? ?F@?i\??!?????Unknown
V&HostSum"Sum_2(1??x?&?@9??x?&?@A??x?&?@I??x?&?@a?a??=?i??f?l????Unknown
`'HostGatherV2"
GatherV2_1(1㥛? ?@9㥛? ?@A㥛? ?@I㥛? ?@aOF?M??=?iA?0?#????Unknown
?(HostDataset">Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1      @9      @A      @I      @a?H/?*=?i*P??????Unknown
?)HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1j?t??@9j?t??@Aj?t??@Ij?t??@a?J??_?;?i$?~8????Unknown
?*HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1????K?@9????K?@A????K?@I????K?@aMt??C?9?i?W't????Unknown
}+HostMul",gradient_tape/sequential/dropout/dropout/Mul(1\???(\@9\???(\@A\???(\@I\???(\@a?K??9?iȤ????Unknown
?,HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast(1?G?z@9?G?z@A?G?z@I?G?z@a ,:???9?i^????????Unknown
?-HostDataset"@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(17?A`??@97?A`??@A7?A`??@I7?A`??@a?j??5?i?5??p????Unknown
?.HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1+?Y@9+?Y@A+?Y@I+?Y@a?q???3?i.????????Unknown
t/HostAssignAddVariableOp"AssignAddVariableOp(1??x?&1@9??x?&1@A??x?&1@I??x?&1@aY%M??2?i?3??9????Unknown
v0HostAssignAddVariableOp"AssignAddVariableOp_4(1??v???@9??v???@A??v???@I??v???@aYlA0?i??F?;????Unknown
Y1HostPow"Adam/Pow(1q=
ףp@9q=
ףp@Aq=
ףp@Iq=
ףp@a??dEp.?iJ?*?"????Unknown
?2HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1??MbX@9??MbX@A??MbX@I??MbX@a???aCX.?i?ar????Unknown
?3HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1X9??v@9X9??v@AX9??v@IX9??v@ay?:*~-?i??U?????Unknown
X4HostEqual"Equal(1?v??/@9?v??/@A?v??/@I?v??/@a?ԍ?@Q*?i?0i?????Unknown
y5HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1+??	@9+??	@A+??	@I+??	@a????#)?i̠Q?????Unknown
v6HostAssignAddVariableOp"AssignAddVariableOp_3(1D?l???@9D?l???@AD?l???@ID?l???@a?B????'?iЏ?֓????Unknown
]7HostCast"Adam/Cast_1(1q=
ףp@9q=
ףp@Aq=
ףp@Iq=
ףp@a???.?'?i?Jpi????Unknown
t8HostReadVariableOp"Adam/Cast/ReadVariableOp(1R???Q@9R???Q@AR???Q@IR???Q@a?COq?'?i<? ?????Unknown
X9HostCast"Cast_2(1??(\??@9??(\??@A??(\??@I??(\??@a???? '?i9+?*?????Unknown
v:HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1????K7@9????K7@A????K7@I????K7@aĚ??y&?iSw8?^????Unknown
?;HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1?v??/@9?v??/@A?v??/@I?v??/@a\????q&?i!???????Unknown
b<HostDivNoNan"div_no_nan_1(1L7?A`?@9L7?A`?@AL7?A`?@IL7?A`?@ai?y??:$?i?l?	????Unknown
a=HostIdentity"Identity(1%??C?@9%??C?@A%??C?@I%??C?@a?A????"?iPlGP8????Unknown?
`>HostDivNoNan"
div_no_nan(1??/?$@9??/?$@A??/?$@I??/?$@any?S?? ?i??|?A????Unknown
u?HostReadVariableOp"div_no_nan/ReadVariableOp(1???S? @9???S? @A???S? @I???S? @a2????Y ?i???wG????Unknown
w@HostReadVariableOp"div_no_nan_1/ReadVariableOp(1?I+? @9?I+? @A?I+? @I?I+? @a#(H  ?i??|G????Unknown
?AHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1=
ףp= @9=
ףp= @A=
ףp= @I=
ףp= @a?????q?i:(?
C????Unknown
oBHostReadVariableOp"Adam/ReadVariableOp(1V-????9V-????AV-????IV-????a??欛?i?^??7????Unknown
vCHostAssignAddVariableOp"AssignAddVariableOp_1(1??C?l??9??C?l??A??C?l??I??C?l??a?arl?iDZ?H+????Unknown
XDHostCast"Cast_3(1P??n???9P??n???AP??n???IP??n???a/{???i?"?????Unknown
vEHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1??Q????9??Q????A??Q????I??Q????a? ?{4??i^??:?????Unknown
TFHostMul"Mul(1??/?$??9??/?$??A??/?$??I??/?$??aШ5?O?i{Dȶ?????Unknown
vGHostCast"$sparse_categorical_crossentropy/Cast(1?z?G???9?z?G???A?z?G???I?z?G???a?G8;?i=>?h?????Unknown
[HHostPow"
Adam/Pow_1(1?~j?t???9?~j?t???A?~j?t???I?~j?t???a?6;????i???E????Unknown
wIHostReadVariableOp"div_no_nan/ReadVariableOp_1(1?5^?I??9?5^?I??A?5^?I??I?5^?I??aJ???H?i?????????Unknown*?C
uHostFlushSummaryWriter"FlushSummaryWriter(1???a??@9???a??@A???a??@I???a??@aK?ʝ5%??iK?ʝ5%???Unknown?
?HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1`??"???@9`??"???@A`??"???@I`??"???@a:[????i?%?wY???Unknown
oHost_FusedMatMul"sequential/dense/Relu(1?I+,?@9?I+,?@A?I+,?@I?I+,?@a?؎L????i??K?y???Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1Zd;?O??@9Zd;?O??@AZd;?O??@IZd;?O??@a??\?kb??i?r??f???Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1?VN?@9?VN?@A?VN?@I?VN?@a?Ɓ?A???i????jw???Unknown
^HostGatherV2"GatherV2(1???Mb?d@9???Mb?d@A???Mb?d@I???Mb?d@a?#?yK??i?\?Q5 ???Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1㥛? xb@9㥛? xb@A㥛? xb@I㥛? xb@a O_???i??'G?x???Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1`??"?`@9`??"?`@A`??"?`@I`??"?`@a??NC??i?{?????Unknown
?	HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(19??v?/Y@99??v?/Y@A9??v?/Y@I9??v?/Y@a9G1y???i(@d?3???Unknown
t
Host_FusedMatMul"sequential/dense_1/BiasAdd(1???Mb@U@9???Mb@U@A???Mb@U@I???Mb@U@a???P??i?k?0?x???Unknown
oHostMul"sequential/dropout/dropout/Mul(1?V?T@9?V?T@A?V?T@I?V?T@a??ر???i[?{w޻???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1ˡE???R@9ˡE???R@AˡE???R@IˡE???R@a???+??~?i?
??????Unknown
rHostDataset"Iterator::Root::ParallelMapV2(1?5^?I?I@9?5^?I?I@A?5^?I?I@I?5^?I?I@a????!,u?iE??Z$???Unknown
?HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(11?Z4H@91?Z4H@A1?Z4H@I1?Z4H@a??s?ic?4??K???Unknown
?Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1NbX9?G@9NbX9?G@ANbX9?G@INbX9?G@a=???#)s?i?-???q???Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1?$??E@9?$??E@A?$??E@I?$??E@a??E)q?i??g?????Unknown
?HostDataset"4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat(1?ZdkB@9?ZdkB@A??(\?b?@I??(\?b?@a???_?i?i??J??????Unknown
?HostDataset".Iterator::Root::ParallelMapV2::Zip[0]::FlatMap(1m????RC@9m????RC@A???S?%?@I???S?%?@aW???`i?i??	?????Unknown
oHostSoftmax"sequential/dense_1/Softmax(1???(\?;@9???(\?;@A???(\?;@I???(\?;@af?Eϰ?f?i???@?????Unknown
qHostCast"sequential/dropout/dropout/Cast(1u?V?:@9u?V?:@Au?V?:@Iu?V?:@ad??? ?e?ih??a]????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1??? ??9@9??? ??9@A??? ??9@I??? ??9@a?\?<
e?i|??g???Unknown
iHostWriteSummary"WriteSummary(1??/?$9@9??/?$9@A??/?$9@I??/?$9@a1;ϛ|d?i??9????Unknown?
cHostDataset"Iterator::Root(1㥛? HQ@9㥛? HQ@A-???'1@I-???'1@a??????[?i????*???Unknown
?HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1/?$??+@9/?$??+@A/?$??+@I/?$??+@a'?u	?yV?i??+s6???Unknown
qHostMul" sequential/dropout/dropout/Mul_1(1?? ?r?*@9?? ?r?*@A?? ?r?*@I?? ?r?*@a?\
5g?U?i?XƦ?@???Unknown
?HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1h??|?5(@9h??|?5(@Ah??|?5(@Ih??|?5(@a???`?S?iN?V?J???Unknown
~HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1??? ??'@9??? ??'@A??? ??'@I??? ??'@a?˩?$?S?i??$??T???Unknown
eHost
LogicalAnd"
LogicalAnd(1???Qx&@9???Qx&@A???Qx&@I???Qx&@a?|???NR?i=y?V?]???Unknown?
HostMul".gradient_tape/sequential/dropout/dropout/Mul_2(1? ?rhQ%@9? ?rhQ%@A? ?rhQ%@I? ?rhQ%@a*?p?^Q?i?1??lf???Unknown
[HostAddV2"Adam/add(1T㥛? %@9T㥛? %@AT㥛? %@IT㥛? %@ao?r??6Q?ik@o???Unknown
wHostDataset""Iterator::Root::ParallelMapV2::Zip(1Zd;?OMU@9Zd;?OMU@A??x?&q#@I??x?&q#@aqJ???O?ir???v???Unknown
Z HostArgMax"ArgMax(1{?G?:"@9{?G?:"@A{?G?:"@I{?G?:"@a??????M?i??^?`~???Unknown
?!HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1J+?"@9J+?"@AJ+?"@IJ+?"@a????yM?i???b?????Unknown
l"HostIteratorGetNext"IteratorGetNext(1??C?l?!@9??C?l?!@A??C?l?!@I??C?l?!@a?2??L?i4?*??????Unknown
v#HostAssignAddVariableOp"AssignAddVariableOp_2(1Zd;?? @9Zd;?? @AZd;?? @IZd;?? @a犪EdeK?i?	<?ɓ???Unknown
V$HostSum"Sum_2(1??x?&?@9??x?&?@A??x?&?@I??x?&?@a??;??I?i?#Z
????Unknown
`%HostGatherV2"
GatherV2_1(1㥛? ?@9㥛? ?@A㥛? ?@I㥛? ?@a??y? I?iBw??J????Unknown
?&HostDataset">Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1      @9      @A      @I      @a?~$?|qH?ib@??f????Unknown
?'HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1j?t??@9j?t??@Aj?t??@Ij?t??@a?5@?;G?ioA??5????Unknown
?(HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1????K?@9????K?@A????K?@I????K?@a?#???E?i8??????Unknown
})HostMul",gradient_tape/sequential/dropout/dropout/Mul(1\???(\@9\???(\@A\???(\@I\???(\@a????>zE?i?hh?????Unknown
?*HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast(1?G?z@9?G?z@A?G?z@I?G?z@a??????E?i??O?U????Unknown
?+HostDataset"@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(17?A`??@97?A`??@A7?A`??@I7?A`??@aQ??q?A?i???7?????Unknown
?,HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1+?Y@9+?Y@A+?Y@I+?Y@a;???>?@?i??sG?????Unknown
t-HostAssignAddVariableOp"AssignAddVariableOp(1??x?&1@9??x?&1@A??x?&1@I??x?&1@aC7d?F??i0?U?????Unknown
v.HostAssignAddVariableOp"AssignAddVariableOp_4(1??v???@9??v???@A??v???@I??v???@a?$??;?i??6?6????Unknown
Y/HostPow"Adam/Pow(1q=
ףp@9q=
ףp@Aq=
ףp@Iq=
ףp@a????؝9?i?/N?j????Unknown
?0HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1??MbX@9??MbX@A??MbX@I??MbX@a(R??҉9?i}g?؛????Unknown
?1HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1X9??v@9X9??v@AX9??v@IX9??v@a??
E?8?i??B!?????Unknown
X2HostEqual"Equal(1?v??/@9?v??/@A?v??/@I?v??/@aiV?%&6?i?~?z????Unknown
y3HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1+??	@9+??	@A+??	@I+??	@a9{?ts5?i?v?????Unknown
v4HostAssignAddVariableOp"AssignAddVariableOp_3(1D?l???@9D?l???@AD?l???@ID?l???@a8
M$}4?io???????Unknown
]5HostCast"Adam/Cast_1(1q=
ףp@9q=
ףp@Aq=
ףp@Iq=
ףp@a??)??3?i,ן????Unknown
t6HostReadVariableOp"Adam/Cast/ReadVariableOp(1R???Q@9R???Q@AR???Q@IR???Q@aG]"??3?ix?/3?????Unknown
X7HostCast"Cast_2(1??(\??@9??(\??@A??(\??@I??(\??@a=??[3?i	]??????Unknown
v8HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1????K7@9????K7@A????K7@I????K7@al???u?2?i???_????Unknown
?9HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1?v??/@9?v??/@A?v??/@I?v??/@a??????2?i??x?????Unknown
b:HostDivNoNan"div_no_nan_1(1L7?A`?@9L7?A`?@AL7?A`?@IL7?A`?@a?f?1?i??dJ?????Unknown
a;HostIdentity"Identity(1%??C?@9%??C?@A%??C?@I%??C?@a_????/?i?6???????Unknown?
`<HostDivNoNan"
div_no_nan(1??/?$@9??/?$@A??/?$@I??/?$@ao?s??+?i??ڙ????Unknown
u=HostReadVariableOp"div_no_nan/ReadVariableOp(1???S? @9???S? @A???S? @I???S? @a?u??+?i^?+R????Unknown
w>HostReadVariableOp"div_no_nan_1/ReadVariableOp(1?I+? @9?I+? @A?I+? @I?I+? @a?????*?ijvE????Unknown
??HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1=
ףp= @9=
ףp= @A=
ףp= @I=
ףp= @ax????v*?i????????Unknown
o@HostReadVariableOp"Adam/ReadVariableOp(1V-????9V-????AV-????IV-????aFF???)?i?Y??D????Unknown
vAHostAssignAddVariableOp"AssignAddVariableOp_1(1??C?l??9??C?l??A??C?l??I??C?l??aN?`??)?imh?W?????Unknown
XBHostCast"Cast_3(1P??n???9P??n???AP??n???IP??n???a5?"H?(?i?*5l????Unknown
vCHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1??Q????9??Q????A??Q????I??Q????a?W6^(?i?-??????Unknown
TDHostMul"Mul(1??/?$??9??/?$??A??/?$??I??/?$??ayW?1M%?i??&?A????Unknown
vEHostCast"$sparse_categorical_crossentropy/Cast(1?z?G???9?z?G???A?z?G???I?z?G???aR58??E$?i2???????Unknown
[FHostPow"
Adam/Pow_1(1?~j?t???9?~j?t???A?~j?t???I?~j?t???a?`?/"$?iH???????Unknown
wGHostReadVariableOp"div_no_nan/ReadVariableOp_1(1?5^?I??9?5^?I??A?5^?I??I?5^?I??a_i?s ?#?i?????????Unknown2Nvidia GPU (Maxwell)